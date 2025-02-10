/**
 * @file 06_cudaError.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-31
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "06_cudaError.cuh"

#include <cuda_runtime.h>

#include <iostream>

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    float *ptr = nullptr;

    // kernel function 是异步的, 因此无法立即检查到是否存在异常
    // kernel_func<<<100, 10>>>(ptr);
    launch_kernel(ptr);
    auto code_ = cudaPeekAtLastError();
    std::cout << cudaGetErrorString(code_) << '\n';

    // 对当前设备的核函数进行同步,等待执行完成,可以发现 kernel function 过程中是否异常
    auto code = cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(code) << '\n';

    // 异常会一直存在,以至于后续的函数都会失败
    float *new_ptr = nullptr;
    auto   code__  = cudaMalloc(&new_ptr, 100);
    std::cout << cudaGetErrorString(code__) << '\n';

    return 0;
}
