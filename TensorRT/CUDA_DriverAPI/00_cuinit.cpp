/**
 * @file 01_checkDeviceError.cpp
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

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    /* brief 对于所有 CUDA driver 函数, 必须调用 cuInit() 进行初始化,
     * 否则其他API调用都会返回 CUDA_ERROR_NOT_INITIALIZED
     */
    // CUresult 类型, 用于接受一些可能得错误代码
    CUresult code = cuInit(0);
    if (code != CUresult::CUDA_SUCCESS)
    {
        const char *err_message = nullptr;
        cuGetErrorString(code, &err_message); /* 获取错误代码的字符串模式 */
        // cuGetErrorName(code, &err_message); /* 也可以直接获取错误代码的字符串 */
        printf("Initialize failed. code = %d, message = %s\n", code, err_message);
        return -1;
    }

    /* brief 测试获取当前 CUDA 的驱动版本
     * 显卡, CUDA, CUDA Toolkit
     * 1. 显卡驱动版本：例如 Driver Version: 551.78
     * 2. CUDA驱动版本: 例如 CUDA Version 12.04
     * 3. CUDA Toolkit 版本: 下载时候所选择的版本 例如 11.2 12.04
     * nvidia-smi 显示的是显卡驱动和此显卡驱动所支持的最高CUDA驱动版本
     * 
     */
    int driver_version = 0;
    code               = cuDriverGetVersion(&driver_version); /* 获取驱动版本 */
    // 120.40 指代 12.4 版本
    printf("CUDA Driver version is %d\n", driver_version);

    char     device_name[100];
    CUdevice device = 0;
    code            = cuDeviceGetName(device_name, sizeof(device_name), device); /* 获取设备名称 */
    // printf("Device %d name is %s\n", device, device_name);
    printf("Device %d name is %s\n code=%d", device, device_name, code);

    return 0;
}
