#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: validate.py
@Python Version: 3.12.1
@Platform: PyTorch 2.0.0 + cu117
@Author: Wei Li (Ithaca)
@Date: 2024-10-04.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

import onnxruntime as ort
import numpy as np


if __name__ == "__main__":
    onnx_file_model = "models/sr.onnx"
    # 找到 GPU / CPU
    provider = ort.get_available_providers()[1 if ort.get_device() == "GPU" else 0]
    print("设备:", provider)
    # 声明 onnx 模型
    model = ort.InferenceSession(onnx_file_model, providers=[provider])

    # 参考: ort.NodeArg
    for node_list in model.get_inputs(), model.get_outputs():
        for node in node_list:
            attr = {"name": node.name, "shape": node.shape, "type": node.type}
            print(attr)
        print("-" * 80)

    # 得到输入、输出结点的名称
    input_node_name = model.get_inputs()[0].name
    output_node_name = [node.name for node in model.get_outputs()]

    image = np.random.random([1, 3, 512, 512]).astype(np.float32)
    output = model.run(
        output_names=output_node_name, input_feed={input_node_name: image}
    )
    print(output[0].shape)
