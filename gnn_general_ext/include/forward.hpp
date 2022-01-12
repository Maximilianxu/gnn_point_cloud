#pragma once

#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <string>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> SAGEForward(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part_nodes,
    int part_size, 
    int dim_worker, 
    int warp_per_block,
    std::string& aggr_type);

std::vector<torch::Tensor> build_part(
    int part_size, 
    torch::Tensor indptr);