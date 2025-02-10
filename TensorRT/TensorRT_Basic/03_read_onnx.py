#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 03_read_onnx.py
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

import onnx
import onnx.helper as helper
import numpy as np


model = onnx.load("./../../output/demo.onnx")


print("============ONNX node info")
# print(helper.printable_graph(model.graph))
print(model)

conv_weight = model.graph.initializer[0]
conv_bias = model.graph.initializer[1]

# 数据是以protobuf格式存储的,因此当中的数值会以bytes的类型存储
# 通过 np.frombuffer 方法还原为类型 float32 的数据类型
print(f"--------------- {conv_weight.name} ---------------")
print(conv_weight.name, np.frombuffer(conv_weight.raw_data, dtype=np.float32))
print(f"--------------- {conv_bias.name} ---------------")
print(conv_bias.name, np.frombuffer(conv_bias.raw_data, dtype=np.float32))
 