/**
 * @file 07_int8_quantization.cpp
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
// onnx解析器头文件
#include <NvOnnxParser.h>

// cuda-runtime include
#include "opencv2/opencv.hpp"

#include <cuda_runtime.h>

// system include
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

inline const char *severity_string(nvinfer1::ILogger::Severity t)
{
    switch (t)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:
        return "error";
    case nvinfer1::ILogger::Severity::kWARNING:
        return "warning";
    case nvinfer1::ILogger::Severity::kINFO:
        return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return "verbose";
    default:
        return "Unkown";
    }
}

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const nvinfer1::AsciiChar *msg) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            /* 打印带颜色的字符,格式如下:
             * printf("\033[47;33m打印的文本\033[0m");
             ? 其中 \033[ 是启始标志
             ?      47 是背景颜色
             ?      ; 是分隔符
             ?      33 是文字颜色
             ?      m 是开始标志结束
             ?      \033[0m 是终止标记
             * 其中背景颜色和文字颜色可以不写
             * 部分颜色代码: https://blog.csdn.net/ericbar/article/details/79652086
             */
            if (severity == Severity::kWARNING)
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            else if (severity <= Severity::kERROR)
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            else
                printf("%s: %s\n", severity_string(severity), msg);
        }
    }
};

bool build_model()
{
    // -----------> step0. TRT-Logger -------------
    TRTLogger logger;

    // -----------> step1. 定义 builder & config & network -------------
    nvinfer1::IBuilder           *pBuilder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig     *pConfig  = pBuilder->createBuilderConfig();
    nvinfer1::INetworkDefinition *pNetwork = pBuilder->createNetworkV2(1);

    // -----------> step2. 输入,模型结构和输出的基本信息 -------------
    // 通过 NvOnnxParser 解析的结果会填充到 Network 中, 类似 addConv 的范式添加进去
    nvonnxparser::IParser *pParser = nvonnxparser::createParser(*pNetwork, logger);
    if (!pParser->parseFromFile("output/demo.onnx", 1))
    {
        printf("Create ONNX parser is NOT successfully\n");
        // NOTE: 需要释放指针, 否则内存泄漏, 优化优雅做法 smart-pointer(TensorRT-V10)
        return false;
    }

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); /* 256MB */
    // 配置暂存寄存器, 用于 layer 实现的临时存储, 也用于存储中间激活值
    pConfig->setMaxWorkspaceSize(1 << 28);

    // -----------> step2.1 关于 Profile -------------
    // 如果模型有多个输入, 则必须有多个 profile
    // 多个输入共用一个 profile
    auto profile      = pBuilder->createOptimizationProfile();
    auto input_tensor = pNetwork->getInput(0);
    int  input_dims   = input_tensor->getDimensions();

    input_dims.d(0) = 1;
    pConfig->setFlag(nvinfer1::BuilderFlag::kINT8);

    auto preProcess
        = [](int current, int count, const std::vector<std::string> &files, nvinfer1::Dims dims, float *ptensor)
    {
        printf("Preprocess %d / %d\n", count, current);

        // 标定所采用的数据预处理必须和推理时一致
        int   width  = dims.d[3];
        int   height = dims.d[2];
        float mean[] = {0.485, 0.456, 0.406};
        float std[]  = {0.229, 0.224, 0.225};

        for (int i{0}; i < files.size(); ++i)
        {
            auto image = cv::imread(files[i]);
            cv::resize(image, image, cv::Size(width, height));
            int            image_area = width * height; // offset for pointer data
            unsigned char *pimage     = image.data;
            float         *phost_b    = ptensor + image_area * 0;
            float         *phost_g    = ptensor + image_area * 1;
            float         *phost_r    = ptensor + image_area * 2;
            for (int i{0}; i < image_area; ++i, pimage += 3)
            {
                // brg --> rgb;
                *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
                *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
                *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
            }
            ptensor += image_area * 3; /* rgb-rgb-rgb ---> rrr-ggg-bbb */
        }
    };

    // 配置 int8 标定数据读取工具
    std::shared_ptr<Int8EntropyCalibrator> calib(
        new Int8EntropyCalibrator({"images/buildCUDAError.png", input_dims, preProcess}));
    pConfig->setInt8Calibrator(calib.get());

    // 配置最小允许: [B, C, H, W]=[1, 1, 3, 3]
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(1, input_channel, 3, 3));
    // 配置最优允许: [B, C, H, W]=[1, 1, 3, 3]
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(1, input_channel, 3, 3));
    // 配置最大允许: [B, C, H, W]=[10, 1, 5, 5]
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));
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

    FILE *file_engine = fopen("output/trt_int8_quantization.engine", "wb");
    fwrite(pModelData->data(), 1, pModelData->size(), file_engine);
    fclose(file_engine);

    // -----------> step4. 序列化 engine -------------
    file_engine     = fopen("calib.txt", "wb");
    auto calib_data = calib->getEntropyCalibratorData();
    fwrite(calib_data.data(), 1, calib_data.size(), file_engine);
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
    auto                   engine_data = load_file("output/trt_int8_quantization.engine");
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

    // -----------> step2. 准备需要进行推理的数据并加载到GPU-device -------------
    auto  image  = cv::imread("images/buildCUDAError.png");
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};

    // 对应 Pytorch 的代码部分
    cv::resize(image, image, cv::Size(width, height));
    int            image_area = image.cols * image.rows; // offset for pointer data
    unsigned char *pimage     = image.data;
    float         *phost_b    = ptensor + image_area * 0;
    float         *phost_g    = ptensor + image_area * 1;
    float         *phost_r    = ptensor + image_area * 2;
    for (int i{0}; i < image_area; ++i, pimage += 3)
    {
        // brg --> rgb;
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }

    int input_batch   = 1;
    int input_channel = 3;
    int input_width   = 224;
    int input_height  = 224;
    int input_numel   = input_batch * input_channel * input_height * input_width;

    float *input_data_host   = nullptr;
    float *input_data_device = nullptr;

    cudaMallocHost(&input_data_host, input_numel * sizeof(float));
    cudaMalloc(&input_data_device, input_numel * sizeof(float));

    cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream);

    const int num_classes = 1000;
    float     output_data_host[num_classes];
    float    *output_data_device = nullptr;
    cudaMalloc(&output_data_device, sizeof(output_data_host));

    // -----------> step3. 执行推理并将结果搬回主机端CPU-host -------------
    // 明确当前推理时, 使用的数据输入大小 shape=[B,C,H,W]
    auto input_dims = pExecution_context->setBindingDimensions(0);
    input_dims.d[0] = input_batch;

    // 用一个指针数组指定 input 和 output 在 GPU-device 的指针
    float *bindings[] = {input_data_device, output_data_device};
    bool   success    = pExecution_context->enqueueV2((void **)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // ----------- output the result -------------
    printf("======== the result via inference phase ========\n");
    float *prob          = output_data_host;
    int    predict_label = std::max_element(prob, prob + num_classes) - prob;
    auto   labels        = load_labels("labels.imagenet.txt");
    auto   predict_name  = labels[predict_label];
    float  confidence    = prob[predict_label];
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);

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
