#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <torch/extension.h>
#include "forward.hpp"
#include "reorder.hpp"
// binding to python

extern std::vector<torch::Tensor> SAGEForwardCuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part_nodes,
    int part_size, 
    int dim_workder, 
    int warp_per_block);

std::vector<torch::Tensor> SAGEForward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part_nodes,
    int part_size, 
    int dim_worker, 
    int warp_per_block){
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part_nodes);

  // std::cout << "calling sage cuda" << std::endl;
  return SAGEForwardCuda(input, weight, row_pointers, column_index, 
                            degrees, part_pointers, part_nodes, 
                            part_size, dim_worker, warp_per_block);
}

std::vector<torch::Tensor> build_part(
    int part_size, 
    torch::Tensor indptr
  ) {

  auto indptr_acc = indptr.accessor<int, 1>();
  int num_nodes = indptr.size(0) - 1;
  int degree, this_num_parts, num_parts = 0;

	for(int i = 0; i < num_nodes; i++){
    degree = indptr_acc[i + 1] - indptr_acc[i];
	  if(degree % part_size == 0)
			this_num_parts = degree / part_size;
    else
			this_num_parts = degree / part_size + 1;
    num_parts += this_num_parts;
	}

  auto part_ptr = torch::zeros(num_parts + 1, torch::kI32);
  auto part_to_nodes = torch::zeros(num_parts, torch::kI32);
	
  int part_counter = 0;
	for(int i = 0; i < num_nodes; i++){
    int degree = indptr_acc[i + 1] - indptr_acc[i];
    if(degree % part_size == 0)
			this_num_parts = degree / part_size ;
    else
			this_num_parts = degree / part_size + 1;

    for (int pid = 0; pid < this_num_parts; pid++){
      int part_beg = indptr_acc[i] + pid * part_size;
      int part_end = part_beg + part_size < indptr_acc[i  + 1]? part_beg + part_size: indptr_acc[i + 1];
      part_ptr[part_counter] = part_beg;
      part_to_nodes[part_counter++] = i;
      //std::cout << i << ", " << pid << ", " << part_counter << ", " << part_beg << "-" << part_end << std::endl;
      if (part_end == indptr_acc[i + 1])
        part_ptr[part_counter] = part_end;
    }
	}
  return {part_ptr, part_to_nodes};
}


PYBIND11_MODULE(gnn_ext, m) {
	m.doc() = "pybind11 sort plugin"; 
	m.def("rabbit_reorder", &rabbit_reorder, "sort param: edge_index");
  m.def("test_torch", &test_torch, "test libtorch c++");
  m.def("build_part", &build_part, "build partition");
  m.def("sage_forward", &SAGEForward, "SAGE (CUDA)");
}
