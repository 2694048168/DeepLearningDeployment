/**
 * @file 02_dynamicShape.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

// custom include

// TensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda-runtime include
#include <cuda_runtime.h>

// system include
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// TensorRT 库日志信息(功能作用类似Error code), 便于知道dll库内部出错调试定位问题
class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const nvinfer1::AsciiChar *msg) noexcept override
    {
        if (severity <= Severity::kVERBOSE) // filter log-level
        {
            printf("%d: %s\n", severity, msg);
        }
    }
};

nvinfer1::Weights make_weights(float *pData, int nCount)
{
    nvinfer1::Weights weights;
    weights.count  = nCount;
    weights.type   = nvinfer1::DataType::kFLOAT;
    weights.values = pData;
    return weights;
}

bool build_model()
{
    // -----------> step0. TRT-Logger -------------
    TRTLogger logger; /* trt-logger 是必要的,用于捕获 warning 和 info等信息 */

    // -----------> step1. 定义 builder & config & network -------------
    /* 这是基础必要的组件
     * 1. 需要 builder 去 build 构建网络; 网络自身有结构,可以有不同的配置;
     * 2. 创建一个构建配置, 指定TensorRT应该如何优化该网络; TensorRT生成的模型只有在特定配置下运行;
     * 3. 创建网络定义, createNetworkV2(1)表示显式batch-size; VERSION>=7, 不建议使用 0 表示非显式;
     */
    nvinfer1::IBuilder           *pBuilder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig     *pConfig  = pBuilder->createBuilderConfig();
    nvinfer1::INetworkDefinition *pNetwork = pBuilder->createNetworkV2(1);

    /* build deep learning network or model
        Network Definition:
        image
          |
        conv(3x3, pad=1) input=1,output=1,bias=True,w=[[1.0, 2.0, 3.1],
                                                      [0.1, 0.1, 0.1],
                                                      [0.2, 0.2, 0.2]]
                                                    b=0.0
          |
        relu
          |
        prob
     */
    // -----------> step2. 输入,模型结构和输出的基本信息 -------------
    const int num_input  = 1; /* the in-channel of input tensor */
    const int num_output = 1; /* the out-channel of output tensor */

    float layer_weights_values[] = {1.0, 2.0, 3.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2}; /* 行优先 */
    float layer_bias_values[]    = {0.0};

    // 输入指定数据的名称,数据类型和完整维度, 将输入层添加到网络
    nvinfer1::ITensor *pInput
        = pNetwork->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));

    // 添加全连接层
    nvinfer1::Weights layer_weights = make_weights(layer_weights_values, 9);
    nvinfer1::Weights layer_bias    = make_weights(layer_bias_values, 1);
    auto layer_conv = pNetwork->addConvolution(*pInput, num_output, nvinfer1::DimsHW(3, 3), layer_weights, layer_bias);
    layer_conv->setPadding(nvinfer1::DimsHW(1, 1));

    // 添加激活层
    auto prob = pNetwork->addActivation(*layer_conv->getOutput(0), nvinfer1::ActivationType::kRELU);

    // 将 prob layer 标记为 output of network
    pNetwork->markOutput(*prob->getOutput(0));

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); /* 256MB */
    // 配置暂存寄存器, 用于 layer 实现的临时存储, 也用于存储中间激活值
    pConfig->setMaxWorkspaceSize(1 << 28);

    // -----------> step2.1 关于 Profile -------------
    // 如果模型有多个输入, 则必须有多个 profile
    auto profile = pBuilder->createOptimizationProfile();

    // 配置最小允许: [B, C, H, W]=[1, 1, 3, 3]
    profile->setDimensions(pInput->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3));
    // 配置最优允许: [B, C, H, W]=[1, 1, 3, 3]
    profile->setDimensions(pInput->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 3, 3));
    // 配置最大允许: [B, C, H, W]=[10, 1, 5, 5]
    profile->setDimensions(pInput->getName(), nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(maxBatchSize, num_input, 5, 5));
    pConfig->addOptimizationProfile(profile);

    // -----------> step3. 生成 engine 模型文件 -------------
    nvinfer1::ICudaEngine *pEngine = pBuilder->buildEngineWithConfig(*pNetwork, *pConfig);
    if (nullptr == pEngine)
    {
        printf("build engine with config is NOT successfully\n");
        return false;
    }

    // -----------> step4. 序列化 engine 模型文件并存储到磁盘文件 -------------
    nvinfer1::IHostMemory *pModelData = pEngine->serialize();

    FILE *file_engine = fopen("output/trt_conv_model.engine", "wb");
    fwrite(pModelData->data(), 1, pModelData->size(), file_engine);
    fclose(file_engine);

    // 卸载顺序按照构建顺序倒序, C++ 的析构顺序
    pModelData->destroy();
    pEngine->destroy();
    pNetwork->destroy();
    pConfig->destroy();
    pBuilder->destroy();

    printf("All Build model or network successfully\n");
    return true;
}

std::vector<unsigned char> load_file(const std::string &engine_filepath)
{
    std::ifstream engineFile(engine_filepath, std::ios::in | std::ios::binary);
    if (!engineFile.is_open())
    {
        std::cout << "Error opening engine file: " << engine_filepath << '\n';
        return {};
    }

    engineFile.seekg(0, engineFile.end);
    size_t fsize = engineFile.tellg();

    std::vector<unsigned char> engineData(fsize);
    if (fsize > 0)
    {
        engineFile.seekg(0, engineFile.beg);
        engineFile.read((char *)&engineData[0], fsize);
    }
    engineFile.close();
    return engineData;
}

void inference()
{
    TRTLogger logger;

    // -----------> step1. 准备模型并加载 -------------
    auto                   engine_data = load_file("output/trt_conv_model.engine");
    // 执行推理前,创建推理runtime接口实例,与builder一样,runtime也需要logger
    nvinfer1::IRuntime    *pRuntime = nvinfer1::createInferRuntime(logger);
    // 将模型读取到 engine_data, 则可以对齐反序列化获取对应的engine
    nvinfer1::ICudaEngine *pEngine = pRuntime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (nullptr == pEngine)
    {
        printf("Deserialize cuda engine is NOT successfully\n");
        pRuntime->destroy();
        return;
    }

    // CUDA context 执行上下文关联
    nvinfer1::IExecutionContext *pExecution_context = pEngine->createExecutionContext();

    cudaStream_t stream = nullptr;
    // 创建 CUDA 流(异步执行), 以确定该 batch 的推理是独立的
    cudaStreamCreate(&stream);
    /* build deep learning network or model
        Network Definition:
        image
          |
        conv(3x3, pad=1) input=1,output=1,bias=True,w=[[1.0, 2.0, 3.1],
                                                      [0.1, 0.1, 0.1],
                                                      [0.2, 0.2, 0.2]]
                                                    b=0.0
          |
        relu
          |
        prob
     */

    // -----------> step2. 准备需要进行推理的数据并加载到GPU-device -------------
    float input_data_host[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 0, 1, 1, 1, -1,
    }; /* batch=2 */
    float *input_data_device = nullptr;

    // 3x3 input ---> 3x3 output  shape[B,C,H,W]
    const int ib = 2; // batch
    const int iw = 3; // W
    const int ih = 3; // H
    float     output_data_host[ib * iw * ih];
    float    *output_data_device = nullptr;

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // -----------> step3. 执行推理并将结果搬回主机端CPU-host -------------
    // 明确当前推理时, 使用的数据输入大小 shape=[B,C,H,W]
    pExecution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
    // 用一个指针数组指定 input 和 output 在 GPU-device 的指针
    float *bindings[] = {input_data_device, output_data_device};
    bool   success    = pExecution_context->enqueueV2((void **)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // ----------- output the result -------------
    printf("======== the result via inference phase ========\n");
    for (int b{0}; b < ib; ++b)
    {
        printf("Batch %d, output_data_host = \n", b);
        for (int i{0}; i < iw * ih; ++i)
        {
            printf("%f, ", output_data_host[b * iw * ih + i]);
            if ((i + 1) % iw == 0)
                printf("\n");
        }
    }

    // -----------> step4. 释放内存,清理资源 -------------
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    pExecution_context->destroy();
    pEngine->destroy();
    pRuntime->destroy();
}

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    if (!build_model())
    {
        printf("Build the model phase is NOT successfully\n");
        return -1;
    }
    inference();

    printf("All done successfully\n");
    return 0;
}
