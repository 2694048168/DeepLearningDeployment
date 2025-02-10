/**
 * @file 06_cudaError.cu
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-31
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "06_cudaError.cuh"

__global__ void kernel_func(float *ptr)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos == 999)
        ptr[999] = 42;
}

__host__ void launch_kernel(float *ptr)
{
    kernel_func<<<100, 10>>>(ptr);
}
