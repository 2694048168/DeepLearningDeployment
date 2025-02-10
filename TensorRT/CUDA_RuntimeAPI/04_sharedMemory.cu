#include <stdio.h>

// ========== 查看静态和动态共享变量的地址 ==========
const size_t    static_shared_memory_num_element = 6 * 1024; /* 6KB */
__shared__ char static_shared_memory[static_shared_memory_num_element];
__shared__ char static_shared_memory2[2];

__global__ void demo1_kernel()
{
    // 静态共享变量和动态共享变量在kernel函数外部/内部
    extern __shared__ char dynamic_shared_memory[];  /* 不能指定大小,有启动kernel指定(参数3) */
    extern __shared__ char dynamic_shared_memory2[]; /* 多个动态共享变量同一地址 */

    printf("The Static shared-memory address = %p\n", static_shared_memory);
    printf("The Static shared-memory2 address = %p\n", static_shared_memory2);
    printf("The Dynamic shared-memory address = %p\n", dynamic_shared_memory);
    printf("The Dynamic shared-memory2 address = %p\n", dynamic_shared_memory2);

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("Running the kernel\n");
}

// ========== 如何给 共享变量赋值 ==========
// 定义共享变量,但是不能给定初始值,必须有线程或其他方式赋值
__shared__ int shared_value1;

__global__ void demo2_kernel()
{
    __shared__ int shared_value2;
    if (threadIdx.x == 0)
    {
        // 在线程索引为 0 的时候, 为 shared value 赋初始值
        if (blockIdx.x == 0)
        {
            shared_value1 = 1314;
            shared_value2 = 66;
        }
        else
        {
            shared_value1 = 520;
            shared_value2 = 88;
        }
    }

    // 等待 block 中所有线程执行到这里(阻塞 2个block中的第一个线程)
    __syncthreads();

    printf("%d.%d shared_value1 = %d[%p], shared_value2 = %d[%p]\n", blockIdx.x, threadIdx.x, shared_value1,
           &shared_value1, shared_value2, &shared_value2);
}

void launch()
{
    demo1_kernel<<<1, 1, 12, nullptr>>>();
    demo2_kernel<<<2, 5, 0, nullptr>>>();
}
