cmake_minimum_required(VERSION 3.8)
project(project)



include_directories("/usr/local/cuda-8.0/include")


find_package(CUDA)
enable_language("CUDA")

set(CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

include_directories(mnist/include)
include_directories(src src/network src/network src/tensor src/tensor src/gpu)
set(SOURCE_FILES
        src/main.cpp
        src/gpu/gpu.cpp
        src/network/activations.cpp
        src/network/activations_gpu.cu
        src/network/layers.cpp
        src/network/layers_gpu.cu
        src/network/neuralnet.cpp
        src/network/neuralnet_gpu.cu
        src/tensor/tensor.cpp
        src/tensor/tensor_gpu.cu)
add_executable(project ${SOURCE_FILES})