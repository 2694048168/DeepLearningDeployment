/**
 * @file 02_asyncStream.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-30
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

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    int device_id = 0;
    CheckDeviceRuntime(cudaSetDevice(device_id));

    cudaStream_t stream = nullptr;
    CheckDeviceRuntime(cudaStreamCreate(&stream));

    const auto SIZE      = 100;
    const auto NUM_BYTES = SIZE * sizeof(float);

    float *memory_device = nullptr;
    CheckDeviceRuntime(cudaMalloc(&memory_device, NUM_BYTES));

    float *memory_host = new float[SIZE];
    memory_host[2]     = 520.1314;

    // 异步复制时,发出指令立即返回,并不等待复制动作的完成
    CheckDeviceRuntime(cudaMemcpyAsync(memory_device, memory_host, NUM_BYTES, cudaMemcpyHostToDevice, stream));

    float *memory_page_locked = nullptr;
    // pinned memory or page locked memory
    CheckDeviceRuntime(cudaMallocHost(&memory_page_locked, NUM_BYTES));
    // 同样不等待复制完成, 但是在流 stream 中排队(stream 中任务有序执行)
    CheckDeviceRuntime(cudaMemcpyAsync(memory_page_locked, memory_device, NUM_BYTES, cudaMemcpyDeviceToHost, stream));

    // 同步操作---wait 统一等待 流队列 中的所有操作结束
    printf("The value is %f\n", memory_page_locked[2]);
    CheckDeviceRuntime(cudaStreamSynchronize(stream));
    printf("The value is %f\n", memory_page_locked[2]);

    // 资源清理
    CheckDeviceRuntime(cudaFreeHost(memory_page_locked));
    CheckDeviceRuntime(cudaFree(memory_device));
    CheckDeviceRuntime(cudaStreamDestroy(stream));
    delete[] memory_host;

    return 0;
}
