/**
 * @file 01_inferEngine.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-31
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
        linear(full connected) input=3,output=2,bias=True,w=[[1.0, 2.0, 0.5],
                                                            [0.1, 0.2, 0.5]]
                                                            b=[0.3, 0.8]]
          |
        sigmoid
          |
        prob
     */
    // -----------> step2. 输入,模型结构和输出的基本信息 -------------
    const int num_input  = 3; /* the in-channel of input tensor */
    const int num_output = 2; /* the out-channel of output tensor */

    float layer_linear_weights_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer_linear_bias_values[]    = {0.3, 0.8};

    // 输入指定数据的名称,数据类型和完整维度, 将输入层添加到网络
    nvinfer1::ITensor *pInput
        = pNetwork->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));

    // 添加全连接层
    nvinfer1::Weights layer_linear_weights = make_weights(layer_linear_weights_values, 6);
    nvinfer1::Weights layer_linear_bias    = make_weights(layer_linear_bias_values, 2);
    // !addFullyConnected will be replaced by addMatrixMultiply + addElementWise;
    // !sinc V8.4 deprecated addFullyConnected API
    auto layer_linear = pNetwork->addFullyConnected(*pInput, num_output, layer_linear_weights, layer_linear_bias);

    // 添加激活层
    auto prob = pNetwork->addActivation(*layer_linear->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

    // 将 prob layer 标记为 output of network
    pNetwork->markOutput(*prob->getOutput(0));

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); /* 256MB */
    pConfig->setMaxWorkspaceSize(1 << 28);
    pBuilder->setMaxBatchSize(1); // batch-size in the inference phase

    // -----------> step3. 生成 engine 模型文件 -------------
    nvinfer1::ICudaEngine *pEngine = pBuilder->buildEngineWithConfig(*pNetwork, *pConfig);
    if (nullptr == pEngine)
    {
        printf("build engine with config is NOT successfully\n");
        return false;
    }

    // -----------> step4. 序列化 engine 模型文件并存储到磁盘文件 -------------
    nvinfer1::IHostMemory *pModelData = pEngine->serialize();

    FILE *file_engine = fopen("output/trt_model.engine", "wb");
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

std::vector<char> load_file(const std::string &engine_filepath)
{
    std::ifstream engineFile(engine_filepath, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Error opening engine file: " << engine_filepath << '\n';
        return std::vector<char>();
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cout << "Error loading engine file: " << engine_filepath << '\n';
        return std::vector<char>();
    }

    return engineData;
}

void inference()
{
    TRTLogger logger;

    // -----------> step1. 准备模型并加载 -------------
    auto                   engine_data = load_file("output/trt_model.engine");
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
        linear(full connected) input=3,output=2,bias=True,w=[[1.0, 2.0, 0.5],
                                                            [0.1, 0.2, 0.5]]
                                                            b=[0.3, 0.8]]
          |
        sigmoid
          |
        prob
     */

    // -----------> step2. 准备需要进行推理的数据并加载到GPU-device -------------
    float  input_data_host[] = {1, 2, 3};
    float *input_data_device = nullptr;

    float  output_data_host[2];
    float *output_data_device = nullptr;

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);
    // 用一个指针数组指定 input 和 output 在 GPU-device 的指针
    float *bindings[] = {input_data_device, output_data_device};

    // -----------> step3. 执行推理并将结果搬回主机端CPU-host -------------
    bool success = pExecution_context->enqueueV2((void **)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // -----------> step4. 释放内存,清理资源 -------------
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    pExecution_context->destroy();
    pEngine->destroy();
    pRuntime->destroy();

    // -----------> step5. 手动推理进行验证 -------------
    const int num_input                     = 3;
    const int num_output                    = 2;
    float     layer_linear_weights_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float     layer_linear_bias_values[]    = {0.3, 0.8};

    printf("======== custom-compute ========\n");
    for (int i{0}; i < num_output; ++i)
    {
        float output_host = layer_linear_bias_values[i];
        for (int j{0}; j < num_input; ++j)
        {
            output_host += layer_linear_weights_values[i * num_output + j] * input_data_host[j];
        }

        // sigmoid
        float prob = 1 / (1 + std::exp(-output_host));
        printf("output_prob[%d] = %f\n", i, prob);
    }
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
