/**
 * @file 03_deviceContext03_deviceContext.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

// CUDA 驱动头文件  cuda.h
#include <cuda.h>
#include <stdio.h>

// 使用有参宏定义检测 cuda driver 的错误
// 并定位程序出错的文件名，行号，错误信息
// 宏中使用 do-while 循环可保证程序的正确性
// 进一步封装,便于函数调用
// 宏定义 #define <宏名> (<参数列表>) <宏体>
#define CheckDeviceDriver(op) __check_cuda_driver((op), #op, __FILE__, __FUNCTION__, __LINE__)

bool __check_cuda_driver(CUresult code, const char *op, const char *file, const char *func, int line)
{
    if (code != CUresult::CUDA_SUCCESS)
    {
        const char *err_name    = nullptr;
        const char *err_message = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_message);
        printf("%s ---> %s:%d %s failed. \n  code = %s, message = %s\n", file, func, line, op, err_name, err_message);
        return false;
    }
    return true;
}

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    CheckDeviceDriver(cuInit(0));

    // 为device创建上下文 context
    CUcontext ctxA   = nullptr; /* typedef struct CUctx_st *CUcontext */
    CUcontext ctxB   = nullptr;
    CUdevice  device = 0;

    // 告知具体指定device的某一块设备地址
    CheckDeviceDriver(cuCtxCreate(&ctxA, CU_CTX_SCHED_AUTO, device));
    CheckDeviceDriver(cuCtxCreate(&ctxB, CU_CTX_SCHED_AUTO, device));
    printf("The context of Device ctxA = %p\n", ctxA);
    printf("The context of Device ctxB = %p\n", ctxB);
    /* context stack 栈结构
        ctxB --- stack-top <--- current_context
        ctxA
        ...
     */

    // 获取上下文 context 信息
    CUcontext current_context = nullptr;
    CheckDeviceDriver(cuCtxGetCurrent(&current_context));
    printf("The current context of Device = %p\n", current_context);

    // ========= 可以使用上下文堆栈对设备管理多个上下文
    // 压入当前context
    CheckDeviceDriver(cuCtxPushCurrent(ctxA));
    CheckDeviceDriver(cuCtxGetCurrent(&current_context));
    printf("The current context of Device = %p\n", current_context);

    // pop the current context
    CUcontext popped_context = nullptr;
    CheckDeviceDriver(cuCtxPopCurrent(&popped_context));
    CheckDeviceDriver(cuCtxGetCurrent(&current_context));
    printf("The popped context of Device popped-CTX = %p\n", popped_context);
    printf("The current context of Device = %p\n", current_context);

    CheckDeviceDriver(cuCtxDestroy(ctxA));
    CheckDeviceDriver(cuCtxDestroy(ctxB));

    // 更加推荐使用 cuDevicePrimaryCtxRetain 获取与设备关联的 context
    // !CUDA-Runtime API 也是基于此, 自动为设备关联一个 context
    CheckDeviceDriver(cuDevicePrimaryCtxRetain(&ctxA, device));
    printf("The context of Device(auto-mode) ctxA = %p\n", ctxA);
    CheckDeviceDriver(cuDevicePrimaryCtxRelease(device));

    return 0;
}
