/**
 * @file 02_checkDeviceErrorAdvanced.cpp
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
#define CheckDeviceError(op) __check_cuda_driver((op), #op, __FILE__, __FUNCTION__, __LINE__)

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
    // 检测 cuda-driver API 初始化, flag==0
    // CheckDeviceError(cuInit(2));
    CheckDeviceError(cuInit(0));

    int driver_version = 0;
    if (!CheckDeviceError(cuDriverGetVersion(&driver_version))) /* 获取驱动版本 */
    {
        printf("Get CUDA driver is NOT successfully\n");
        return -1;
    }
    printf("CUDA Driver version is %d\n", driver_version);

    char     device_name[100];
    CUdevice device = 0;
    if (!CheckDeviceError(cuDeviceGetName(device_name, sizeof(device_name), device))) /* 获取设备名称 */
    {
        printf("Get GPU-device name is NOT successfully\n");
        return -1;
    }
    printf("Device %d name is %s\n", device, device_name);

    return 0;
}
