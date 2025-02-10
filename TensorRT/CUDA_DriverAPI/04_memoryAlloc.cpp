/**
 * @file 04_memoryAlloc.cpp
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

    // Create context for a device
    CUcontext context = nullptr;
    CUdevice  device  = 0;
    // !必须要有上下文 context, 否则后续操作都将失败
    if (!CheckDeviceDriver(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device)))
    {
        printf("Create context of Device is NOT successfully\n");
        return -1;
    }
    printf("Create context of Device CTX-pointer = %p\n", context);

    // 输入 device-ptr 向设备申请 100 bytes 的线性内存, 并返回地址
    // ?仔细想想 OpenGL, 就能知道为啥这里指向设备的内存地址的指针是 编号int?
    CUdeviceptr device_memory_pointer = 0;
    if (!CheckDeviceDriver(cuMemAlloc(&device_memory_pointer, 100)))
    {
        printf("Create memory of Device is NOT successfully\n");
        return -1;
    }
    printf("Create memory of Device memory address-pointer = %p\n", (void *)device_memory_pointer);

    // 输入二级指针指向 host 要 100 bytes 的 "索页内存" , 专供设备使用
    // ! 'malloc' maybe not the page-locked memory on HOST
    float *host_page_locked_memory = nullptr;
    if (!CheckDeviceDriver(cuMemAllocHost((void **)&host_page_locked_memory, 100)))
    {
        printf("Create page-locked memory of Host is NOT successfully\n");
        return -1;
    }
    printf("Create page-locked memory of Host address-pointer = %p\n", host_page_locked_memory);

    // 向 host_page_locked_memory 写入数据(CPU), GPU-device quick access
    host_page_locked_memory[0] = 42;
    printf("host_page_locked_memory[0] = %f\n", host_page_locked_memory[0]);

    /*  host_page_locked_memory 声明的时候为 float*, 可以直接转换为 device-ptr,
     * 这样才能直接送给 cuda-kernel function, 以及和设备内存直接转换
     ? CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);
     * 初始化内存的值, 初始化值必须为无符号整型,因此需要和 new_value 进行数据转换:
     ? 但是不能直接写为: (int)value, 必须写为: *(int*)&new_value
     * ----> &new_value 获取 float new_value 的地址;
     * ----> (int*) 将地址从 float* 转为 int* 以避免x64架构上的精度损失;
     * ----> *(int*) 取消引用地址, 最后获取引用的int值;
     */
    float new_value = 666.f;
    CheckDeviceDriver(cuMemsetD32((CUdeviceptr)host_page_locked_memory, *(int *)&new_value, 1));
    printf("host_page_locked_memory[0] = %f\n", host_page_locked_memory[0]);

    // free memory
    CheckDeviceDriver(cuMemFreeHost(host_page_locked_memory));
    // context 必须最后一个销毁动作, 要不然会报错(其他释放操作已经没有上下文了)
    CheckDeviceDriver(cuCtxDestroy(context));

    return 0;
}
