#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 03_edit_onnx.py
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

# 可以获取权重
conv_weight = model.graph.initializer[0]
conv_bias = model.graph.initializer[1]
# 修改权重
conv_weight.raw_data = np.arange(9, dtype=np.float32).tobytes()

# 修改权重后存储
onnx.save_model(model, "./../../output/demo_changedWeight.onnx")
print("All done is successfully\n")
