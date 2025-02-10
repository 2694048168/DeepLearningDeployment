#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 03_gen_onnx_pytorch.py
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
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import os


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x  = self.conv(x)
        x = self.relu(x)
        return x

# 这个包对应 opset11 的导出代码,如果想修改导出的细节,可以在这里修改代码
# import torch.onnx.symbolic_opset11
print("对应的opset文件节路径: ",os.path.dirname(torch.onnx.__file__))

model = Model()
dummy = torch.zeros(1, 1, 3, 3)
torch.onnx.export(model,
                  (dummy,), #这里的args,是指输入给model的参数,需要传递 tuple,因此用括号
                  "./../../output/demo.onnx", #存储文件的文件路径
                  verbose=True, #打印输出详细信息
                  input_names=["image"], #为输入和输出节点指定名称
                  output_names=["output"], #便于后续查看或者操作
                #   opset_version=11, #这里的opset版本,指定各类算子以何种方式导出
                  opset_version=18, #这里的opset版本,指定各类算子以何种方式导出
                  dynamic_axes={ #表示batch,height,width channel的维度是否是动态,ONNX指定-1
                      #通常shape[B,C,H,W], 只设置 B 为动态,其他维度避免动态
                      "image": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}
                  }
)

print("All done is successfully\n")
