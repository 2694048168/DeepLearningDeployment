/**
 * @file 04_sharedMemory.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-31
 * 
 * @copyright Copyright (c) 2024
 * 
 */

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

void launch();

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    int device = 0;

    cudaDeviceProp prop;
    CheckDeviceRuntime(cudaGetDeviceProperties(&prop, device));
    printf("The shared memory: prop.sharedMemoryBlock = %.2f kB\n", prop.sharedMemPerBlock / 1024.0f);
    // 越近越快, 越近越贵

    launch();
    CheckDeviceRuntime(cudaPeekAtLastError());
    CheckDeviceRuntime(cudaDeviceSynchronize());

    printf("All work done\n");
    return 0;
}
