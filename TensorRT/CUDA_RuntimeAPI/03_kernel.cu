/**
 * @file 03_kernel.cu
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__device__ __host__ float sigmod(float x)
{
    return 1 / (1 + exp(-x));
}

__global__ void test_print_kernel(const float *pData, int ndata)
{
    int   idx = threadIdx.x + blockIdx.x * blockDim.x; /* built-in var. */
    /*    dims              indexes
     *  gridDim.z         blockIdx.z
     *  gridDim.y         blockIdx.y
     *  gridDim.x         blockIdx.x
     *  blockDim.z        threadIdx.z
     *  blockDim.y        threadIdx.y
     *  blockDim.x        threadIdx.x
     *
     * Pseudo Code(由高维到低维, 左乘右加):
     * position = 0
     * for i in 6:
     *    position *= dims[i]
     *    position += indexes[i]
     * 
     */
    float y = sigmod(0.5f);
    printf("The result of Sigmod(__device__) = %f\n", y);

    printf("Element[%d] = %f, threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d\n", idx, pData[idx], threadIdx.x, blockIdx.x,
           blockDim.x);
}

__host__ void test_print(const float *pData, int ndata)
{
    float y = sigmod(0.5f);
    printf("The result of Sigmod(__host__) = %f\n", y);

    // <<<gridDim, blockDim, bytes_of_shared_memory, stream>>>
    test_print_kernel<<<1, ndata, 0, nullptr>>>(pData, ndata);

    /* 在核函数kernel执行结束后, 通过 cudaPeekAtLastError 获取得到的错误码, 判断是否出现错误
     * cudaPeekAtLastError(); & cudaGetLastError(); 都可以获取kernel function exec. 错误码
     * cudaGetLastError 是获取错误码并清除,下一次执行cudaGetLastError获取的是success;
     * cudaPeekAtLastError 是获取当前错误码,下一次执行cudaGetLastError or cudaPeekAtLastError 拿到的是上次错误码;
     * CUDA的错误码会传递,如果这里错误了,不移除,那么后续任意API的返回值都会是这个错误码,都会失败.
     */
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        const char *err_name    = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("Kernel function error %s ---> %s:%d test_print_kernel failed.\n  code = %s, message = %s", __FILE__,
               __FUNCTION__, __LINE__, err_name, err_message);
    }
}
