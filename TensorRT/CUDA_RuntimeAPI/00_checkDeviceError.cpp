/**
 * @file 00_checkDeviceError.cpp
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

// CUDA 驱动头文件  cuda.h
#include <cuda.h>
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
    CUcontext context = nullptr;
    cuCtxGetCurrent(&context);
    printf("The current context of Device CTX=%p\n", context);

    /* cuda-runtime API 是以 CUDA 为基准开发的运行时库
     * cuda runtime 所用的 context 是基于 cuDevicePrimaryCtxRetain 函数获取的
     * cuda runtime 是采用懒加载范式, 即 cuda驱动的init和destroy, context管理都封装隐藏
     */
    int device_count = 0;
    CheckDeviceRuntime(cudaGetDeviceCount(&device_count));
    printf("The count of device on the system : %d\n", device_count);

    // 取而代之, 使用 setdevice 来控制当前上下文,
    // 通过不同的 device id 来使用不同设备
    // !NOTE: context 是线程内作用的, 其他线程不相关的, 一个线程一个 context stack
    int device_id = 0;
    printf("Set current device to : %d, the context tagged\n", device_id);
    CheckDeviceRuntime(cudaSetDevice(device_id));

    cuCtxGetCurrent(&context);
    printf("The current context of Device CTX=%p\n", context);

    int current_device = -1;
    CheckDeviceRuntime(cudaGetDevice(&current_device));
    printf("The current Device=%d\n", current_device);

    return 0;
}
