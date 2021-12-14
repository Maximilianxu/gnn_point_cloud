cmake_minimum_required(VERSION 3.10)
project(gnn_ext CXX CUDA)

set(VERSION 0.0.1)
set(CXX_FLAGS ${CMAKE_CXX_FLAGS} -mcx16 -O3  -ltcmalloc_minimal -lnuma)
set(CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --compiler-options '-fPIC' -gencode=arch=compute_70,code=sm_70 --expt-relaxed-constexpr)

message(STATUS "project name: ${PROJECT_NAME}, version: ${VERSION}")

# some global sets
set(TARGET_INC ${CMAKE_CURRENT_LIST_DIR}/../include)
set(TORCH_PATH /root/miniconda3/envs/pyg/lib/python3.7/site-packages/torch)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(PYBIND11_CT _gcc)
set(PYBIND11_STD _libstdcpp)
set(PYBIND11_BUILD_ABI _cxxabi1011)

option(CXX17 "use c++ 17 or not" OFF)

add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)
add_definitions(-DPYBIND11_COMPILER_TYPE="${PYBIND11_CT}")
add_definitions(-DPYBIND11_STDLIB="${PYBIND11_STD}")
add_definitions(-DPYBIND11_BUILD_ABI="${PYBIND11_BUILD_ABI}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

file(GLOB TARGET_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/../src/*.cpp)
file(GLOB CUDA_KERNELS CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/../src/cuda/*.cu)
# add_subdirectory(cuda/)

# basic libs
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

# NOTE: we use torch_ext compile info to link, not libtorch (which may incur mismatched versions)
# -L/root/miniconda3/envs/pyg/lib/python3.7/site-packages/torch/lib -L/usr/local/cuda-11.2/lib64 
# -L/usr/local/cuda/lib64/libcudnn.so.7/lib64 -lc10 -ltorch 
# -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda_cu -ltorch_cuda_cpp
link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(${TORCH_PATH}/lib /usr/local/cuda-11.2/lib64 /usr/local/cuda/lib64/libcudnn.so.7/lib64)
link_directories(${CMAKE_CURRENT_LIST_DIR}/../src/cuda/)
include_directories(${TARGET_INC} /usr/include ${TORCH_PATH}/include ${TORCH_PATH}/include/torch/csrc/api/include
  ${TORCH_PATH}/TH PRIVATE ${TORCH_PATH}/THC /usr/local/cuda-11.2/include /usr/local/cuda/lib64/libcudnn.so.7/include
  /root/miniconda3/envs/pyg/include/python3.7m)

add_library(cuda_impl OBJECT ${CUDA_KERNELS})
# CUDA_COMPILE(cuda_impl ${CUDA_KERNELS})
# $<TARGET_OBJECTS:cuda_impl>
add_library(${PROJECT_NAME} MODULE ${TARGET_SRC} $<TARGET_OBJECTS:cuda_impl>)

# basic props
if(CXX17)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
  set_property(TARGET cuda_impl PROPERTY CUDA_STANDARD 17)
else()
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
  set_property(TARGET cuda_impl PROPERTY CUDA_STANDARD 14)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
set_target_properties(${PROJECT_NAME} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

target_compile_options(cuda_impl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS} ${Threads_CXX_FLAGS} ${OpenMP_CXX_FLAGS}>)
# add_compile_options(${CXX_FLAGS})
# target_include_directories(${PROJECT_NAME} PRIVATE ${TARGET_INC})

# -I/root/miniconda3/envs/pyg/lib/python3.7/site-packages/torch/include 
# -I/root/miniconda3/envs/pyg/lib/python3.7/site-packages/torch/include/torch/csrc/api/include 
# -I/root/miniconda3/envs/pyg/lib/python3.7/site-packages/torch/include/TH 
# -I/root/miniconda3/envs/pyg/lib/python3.7/site-packages/torch/include/THC 
# -I/usr/local/cuda-11.2/include 
# -I/usr/local/cuda/lib64/libcudnn.so.7/include -I/root/miniconda3/envs/pyg/include/python3.7m
# target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_PATH}/include PRIVATE ${TORCH_PATH}/include/torch/csrc/api/include)
# target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_PATH}/TH PRIVATE ${TORCH_PATH}/THC)
# target_include_directories(${PROJECT_NAME} PRIVATE /usr/local/cuda-11.2/include PRIVATE 
#   /usr/local/cuda/lib64/libcudnn.so.7/include PRIVATE /root/miniconda3/envs/pyg/include/python3.7m)

#####################################
# find_package(Torch REQUIRED PATHS ${TORCH_PATH}) # deprecated libtorch and link ${TORCH_LIBRARIES}, use torch_ext instead
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#####################################

target_link_libraries(${PROJECT_NAME} PRIVATE tcmalloc_minimal numa Threads::Threads OpenMP::OpenMP_CXX pybind11::module pybind11::lto pybind11::windows_extras
  c10 torch torch_cpu torch_python cudart c10_cuda torch_cuda_cu torch_cuda_cpp)

pybind11_extension(${PROJECT_NAME})
pybind11_strip(${PROJECT_NAME})
