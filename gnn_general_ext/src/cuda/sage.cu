#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <torch/extension.h>
#include <stdio.h>

#define WARP_SIZE 32

__global__ void warmup(){}

__device__ inline 
void atomicAddF(float* address, float value){
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f);
}

template <typename scalar_t>
__global__ void cudaGatherReduce(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_nodes,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int part_size,
    const int dim_worker,
    const int warp_per_block
){  
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpid = tid / WARP_SIZE;                             // global warp-id
    int block_warpid = threadIdx.x / WARP_SIZE;               // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                      // part information.
    int *partial_ids = part_meta;                                           // caching ids
    float *partial_results = (float*)&part_meta[part_size * warp_per_block];     // caching partial results.

    if (warpid < num_parts){

        int srcid = part_nodes[warpid];              // aggregated target node
        int part_beg = part_pointers[warpid];        // partitioning pointer start
        int part_end = part_pointers[warpid + 1];    // part pointer end
        float src_norm = degrees[srcid];            // norm of the target node

        // Cache the part neighbors by all threads from a warp.
        const int pindex_base = block_warpid * part_size;
        #pragma unroll
        for (int nidx = part_beg + laneid; nidx < part_end; nidx += WARP_SIZE){
            // if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
            partial_ids[pindex_base + nidx - part_beg] = column_index[nidx];
        }
        
        // #pragma unroll
        // for (int nidx = part_beg; nidx < part_end; nidx++){
        // //     if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
        //     partial_ids[nidx - part_beg] = column_index[nidx];
        // }
        
        __syncwarp();

        // if (laneid == 0)
        // for (int nidx = laneid; nidx < part_end - part_beg; nidx++){
            // int nid = partial_ids[pindex_base + nidx];
            // int nid = partial_ids[nidx];
            // printf("verify nid - 111111: %d\n", nid);
            // if(nid >= num_nodes || nid < 0) printf("verify nid: %d\n", nid);
        // }

        // Neighbor aggregation within each part
        const int presult_base = block_warpid * dim;
        for (int nidx = 0; nidx < part_end - part_beg; nidx++){ // this part is serial
            int nid = partial_ids[pindex_base + nidx];
            // int nid = partial_ids[nidx];
            // if (laneid == 0)
            //     printf("verify nid - 222222: %d\n", nid);
            float degree_norm_inv = __fmaf_rn(src_norm, degrees[nid], 0);

            // Initialize shared memory for partial results
            if (nidx == 0)
                if (laneid < dim_worker)
                    #pragma unroll
                    for (int d = laneid; d < dim; d += dim_worker){
                        partial_results[presult_base + d] = 0.0f;
                    }
            
            if (laneid < dim_worker)
                #pragma unroll
                for (int d = laneid; d < dim; d += dim_worker){
                    // if(nid >= num_nodes || nid < 0) printf("aggregation: %d\n", nid);
                    partial_results[presult_base + d] += __fmaf_rn(1.0, input[nid][d], 0);
                    // partial_results[presult_base + d] += input[nid][d];
                }
        }

        // output the result to global memory from the shared memory
        if (laneid < dim_worker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dim_worker){
                atomicAddF((float*)&output[srcid][d], partial_results[presult_base + d]);
            }
    }
}


std::vector<torch::Tensor> SAGEForwardCuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part_nodes,
    int part_size, 
    int dim_workder, 
    int warp_per_block
) {
    // printf("kernel start\n");
    auto tmp = torch::mm(input, weight);
    // auto output = torch::zeros_like(tmp);
    auto output = torch::zeros({input.size(0), weight.size(1)}, torch::kCUDA);
    const int dim = output.size(1);
    const int num_nodes = output.size(0);
    const int num_parts = part_nodes.size(0);

    const int block = warp_per_block * WARP_SIZE;
    const int grid = (num_parts * WARP_SIZE + block  - 1) / block; 
    int shared_memory = part_size*warp_per_block*sizeof(int)+warp_per_block*dim*sizeof(float);

    // printf("grid: %d, block: %d\n", grid, block);
    // printf("dim: %d, num_nodes: %d, num_parts: %d\n", dim, num_nodes, num_parts);
    // printf("input: (%d, %d)\n", tmp.size(0), tmp.size(1));
    // printf("dim_workder: %d\n", dim_workder);
    // printf("shared_memory: %d\n", tmp.size(0), tmp.size(1));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "sage_forward_cuda", ([&] {
                                cudaGatherReduce<scalar_t><<<grid, block, shared_memory>>>(
                                    output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    tmp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part_nodes.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts,
                                    part_size,
                                    dim_workder,
                                    warp_per_block
                                );
                            }));
                                 
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output};
}
