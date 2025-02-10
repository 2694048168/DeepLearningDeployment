/**
 * @file 05_warpAffine.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-31
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "opencv2/opencv.hpp"

#include <cuda_runtime.h>
#include <stdio.h>

#include <filesystem>

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

// cv::Mat is shallow-copy default, so the 'reference' ...
cv::Mat warpAffine_to_centerAlign(const cv::Mat &srcImg, const cv::Size &size)
{
    return cv::Mat();
}

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::string imgFilepath = R"(images/buildCUDAError.png)";
    std::string saveFolder  = R"(output/)";
    if(!std::filesystem::exists(saveFolder))
        std::filesystem::create_directories(saveFolder);

    cv::Mat srcImg = cv::imread(imgFilepath, cv::IMREAD_UNCHANGED);
    cv::Mat dstImg = warpAffine_to_centerAlign(srcImg, cv::Size(640, 640));

    if (!dstImg.empty())
        cv::imwrite(saveFolder + "output.png", dstImg);
    printf("All Done. save the output.png\n");

    return 0;
}
