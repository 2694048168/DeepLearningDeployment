/**
 * @file 03_kernelFunction.cpp
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

void test_print(const float *pData, int ndata);

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    float     *parray_host   = nullptr;
    float     *parray_device = nullptr;
    const auto SIZE_ARRAY    = 10;
    const auto NUM_BYTES     = sizeof(float) * SIZE_ARRAY;

    parray_host = new float[SIZE_ARRAY];
    CheckDeviceRuntime(cudaMalloc(&parray_device, NUM_BYTES));

    for (int idx{0}; idx < SIZE_ARRAY; ++idx)
    {
        parray_host[idx] = idx;
    }

    CheckDeviceRuntime(cudaMemcpy(parray_device, parray_host, NUM_BYTES, cudaMemcpyHostToDevice));
    test_print(parray_device, SIZE_ARRAY);
    CheckDeviceRuntime(cudaDeviceSynchronize());

    CheckDeviceRuntime(cudaFree(parray_device));
    delete[] parray_host;

    return 0;
}
