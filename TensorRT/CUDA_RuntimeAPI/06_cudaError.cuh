/**
 * @file 06_cudaError.cuh
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-31
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <cuda_runtime.h>

__global__ void kernel_func(float *ptr);

__host__ void launch_kernel(float *ptr);
