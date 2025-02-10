/**
 * @file 01_memoryAlloc.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

// CUDA 运行时头文件  cuda_runtime.h
#include <cuda_runtime.h>
#include <stdio.h>

#define CheckDeviceRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __FUNCTION__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, const char *func, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name    = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("%s ---> %s:%d %s failed. \n  code = %s, message = %s\n", file, func, line, op, err_name, err_message);
        return false;
    }
    return true;
}

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    int device_id = 0;
    CheckDeviceRuntime(cudaSetDevice(device_id));

    const auto NUM_BYTES = 100 * sizeof(float);

    float *memory_device = nullptr;
    // pointer to device global memory
    CheckDeviceRuntime(cudaMalloc(&memory_device, NUM_BYTES));

    float *memory_host = new float[100]; /* pageable memory */
    memory_host[2]     = 520.1314;
    CheckDeviceRuntime(cudaMemcpy(memory_device, memory_host, NUM_BYTES, cudaMemcpyHostToDevice));

    float *memory_page_locked = nullptr;
    // pinned memory or page locked memory
    CheckDeviceRuntime(cudaMallocHost(&memory_page_locked, NUM_BYTES));
    CheckDeviceRuntime(cudaMemcpy(memory_page_locked, memory_device, NUM_BYTES, cudaMemcpyDeviceToHost));
    printf("The value is %f\n", memory_page_locked[2]);

    CheckDeviceRuntime(cudaFreeHost(memory_page_locked));
    delete[] memory_host;
    CheckDeviceRuntime(cudaFree(memory_device));

    return 0;
}
