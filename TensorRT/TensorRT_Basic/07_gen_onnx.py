#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 07_gen_onnx.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-01-01.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 
'''

import torch
import torchvision
import cv2
import numpy as np


class Classifier(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = torchvision.models.resnet18(weights=True)

    def forward(self, x):
        feature = self.backbone(x)
        # 后处理能够在 onnx 里面处理就更好, C++ 不需要自己写后处理代码
        probability = torch.softmax(feature, dim=1)
        return probability


# ------------------------------
if __name__ == "__main__":
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    image = cv2.imread("./../../images/buildCUDAError.png")
    # ! warpAffine via CUDA kernel to acc.
    image = cv2.resize(image, (224, 224)) #pre-processing: resize
    image = image[..., ::-1] #pre-processing: BGR ---> RGB
    image = image / 255.0
    image = (image - imagenet_mean) / imagenet_std #pre-processing: normalize
    image = image.astype(np.float32) #pre-processing: float64 ---> float32
    image = image.transpose(2, 0, 1) #pre-processing: HWC ---> CHW
    image = np.ascontiguousarray(image) #contiguous array memory 
    image = image[None, ...] #pre-processing: toTensor shape: CHW ---> 1CHW
    image = torch.from_numpy(image) # numpy ---> torch

    model = Classifier().eval()
    with torch.no_grad():
        probability = model(image)

    predict_class = probability.argmax(dim=1).item()
    confidence = probability[0, predict_class]
    # labels = open("labels.imagenet.text").readlines()
    # labels = [item.strip() for item in labels]
    # print(f"Predict: {predict_class}, {confidence}, {labels[predict_class]}")
    print(f"Predict: {predict_class}, {confidence}")

    # ------- export onnx -------
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, (dummy,), 
                      "./../../output/int8_quantization.onnx",
                      input_names=["image"],
                      output_names=["prob"],
                      dynamic_axes={"image": {0: "batch"}, "prob": {0: "batch"}},
                      opset_version=17
                      )

    print("All done is successfully\n")
