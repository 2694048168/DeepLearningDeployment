## MNIST, the HelloWold for Model Deployment 

> MNIST数据库是一个通常用于训练各种数字图像处理系统的大型数据库, 该数据库通过对来自NIST原始数据库的样本进行修改创建, 涵盖手写数字的图像, 共包含60,000张训练图像和10,000张测试图像, 尺寸为28×28像素; 该数据库广泛运用于机器学习领域的训练与测试当中.

### Quick Start

```shell
# Python code and inference
git clone ModelDeployment
cd ModelDeployment

# C++ code and inference

```

### Organization of Source Code

```
. ModelDeployment
|—— MNIST
|   |—— data
|   |   |—— mnist
|   |   |   |—— MNIST
|   |—— models
|   |   |—— model_best.pth
|   |   |—— model.onnx
|   |   |—— model_Engine.trt
|   |   |—— checkpoints
|   |   |   |—— epoch_1_ckp.pth
|   |   |   |—— epoch_2_ckp.pth
|   |   |   |—— iteration_100_ckp.pth
|   |   |   |—— iteration_500_ckp.pth
|   |   |   |—— iteration_900_ckp.pth
|   |—— logs
|   |   |—— rootLog.log
|   |   |—— trainLog.log
|   |   |—— testLog.log
|   |   |—— inferenceLog.log
|   |—— networks.py
|   |—— train.py
|   |—— test.py
|   |—— pytorch_export_onnx.py
|   |—— onnx_export_trtEngine.py
|   |—— infer_tensorrt_onnx.py
|   |—— infer_tensorrt_onnx.cpp
|   |—— requirements.txt
|   |—— CMakeLists.txt
|   |—— README.md
```
