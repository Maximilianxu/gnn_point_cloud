# gnn_point_cloud

## Intro

This repo is used for me to learn gnn on point clouds.

Basic parts in this repo contain the implementation for paper
```
@InProceedings{Point-GNN,
author = {Shi, Weijing and Rajkumar, Ragunathan (Raj)},
title = {Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
, but using native Torch, pyg, and DGL. Their original implementation is based on TF1.5.

Also, I will extend this repo by C++ soon.

## Directory Description
*.py in root dir is used to benchmark pyg/DGL/native Torch for GNNs.

conf/ contains several yamls to configure dataset, graph generation, network architectures

gnn_general_ext/ this is what I used to extend Py with C++/CUDA, depending on libtorch basically

gnn_advisor_ext/ this is GNNAdvisor (OSDI 2021), I just copy most of its codes here

## How to install

Benching pyg/dgl only requires installation of PyG and DGL. Nothing else.

For gnn_general_ext, it contains an extension named gnn_ext, just executing
```
./build.sh
```
will be fun. This script will build and install an extension named gnn_ext, which can be tested at tests/*.py (ref. these tests for its use).

For gnn_advisor_ext, just running
```
./build.sh 
```
will be fun.

## Some Pitfalls
### Why didn't I use the recommended torch_extension toolchains to build the C++ extension directly?
It is too slow. I can't stand with it. So, I still choose to use cmake, that I am familiar with.

### About Cmake
I use cmake 3.10, although I used an older version before, which is **very different** from this version.

As far as I know, cmake experienced some major refactoring during 3.10-3.12(maybe 3.10-3.18).

1. target_compile_options is recommended to append flags, rather than the old set(CMAKE_CXX_FLAGS xxx)
2. target_link_libraries is recommended to link libs
3. set_target_properties is recommended to set CXX/CUDA standard, rather than set(CXX_FLAGS, -std=c++14)
4. The following snippet is recommended to link cuda objects and c++
```
add_library(cuda_impl OBJECT ${CUDA_KERNELS})
add_library(${PROJECT_NAME} MODULE ${TARGET_SRC} $<TARGET_OBJECTS:cuda_impl>)
```
rather than 
```
CUDA_COMPILE(cuda_impl ${CUDA_KERNELS})
add_library(${PROJECT_NAME} MODULE ${TARGET_SRC} ${cuda_impl})
```
why? because we can control the compilation of cuda and c++ separately by the first way. For example,
```
set_property(TARGET cuda_impl PROPERTY CUDA_STANDARD 17)
target_compile_options(cuda_impl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
```
### About PyTorch and Libtorch
Although I didn't use the torch_extension toolchains to build, I still rely on torch_extension to write the codes. Why?

Because I want to import them using Python as the frontend, and using libtorch may incur mismatched versions with pytorch. Meanwhile, using torch_extension can link the torch_python.so directly.

### Other Pitfalls (just recording)
I have some typos in the original cmake. I wanted to write *.cpp to specify source files, but I wrote main.cpp actually. Then the python frontend complains some symbols like "Nbuild_partExxx" are missing. Of course, I know this is C++ name mangling. And the original function name should be "build_part", which indicats missed implementation files (should change main.cpp to *.cpp). However, I thought this was due to the ABI mismatch error and wasted much time, because I didn't look at it carefully!

Another error is because the two extensions (gnn_general_ext and gnn_advisor_ext) have similar names, so as the libs. I wasted so much time just because I didn't notice that the error was from another lib... I was blind :(.