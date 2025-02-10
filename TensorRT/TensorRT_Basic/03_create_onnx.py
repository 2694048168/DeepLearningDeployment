#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 03_create_onnx.py
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


# https://github.com/onnx/onnx/blob/main/onnx/onnx-ml.proto3
nodes = [
    helper.make_node(
        name="Conv_0", #节点名称, 不要和 op_type 搞混了
        op_type="Conv", #节点的算子类型, 如 'Conv' | 'Relu' | 'Add', 详细查看onnx算子列表
        inputs=["image", "conv.weight", "conv.bias"], #各个输出的名称,节点输入包括:输入和算子权重
        outputs=["3"],
        pads=[1, 1, 1, 1],
        group=1,
        dilations=[1, 1],
        kernel_shape=[3, 3],
        stride=[1, 1]
    ),
    helper.make_node(
        name="ReLu_1",
        op_type="Relu",
        inputs=["3"],
        outputs=["output"]
    )
]

initializer = [
    helper.make_tensor(
        name="conv.weight",
        data_type=helper.TensorProto.DataType.FLOAT,
        dims=[1, 1, 3, 3],
        vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).tobytes(),
        raw=True
    ),
    
    helper.make_tensor(
        name="conv.bias",
        data_type=helper.TensorProto.DataType.FLOAT,
        dims=[1],
        vals=np.array([0.0], dtype=np.float32).tobytes(),
        raw=True
    ),
]

inputs = [
    helper.make_value_info(
        name="image",
        type_proto=helper.make_tensor_type_proto(
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    )
]

outputs = [
    helper.make_value_info(
        name="output",
        type_proto=helper.make_tensor_type_proto(
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    )
]

graph = helper.make_graph(
    name="mymodel",
    inputs=inputs,
    outputs=outputs,
    nodes=nodes,
    initializer=initializer
)

# 如果名字不是 'ai.onnx', netron解析就不是太一样
opset = [
    # helper.make_operatorsetid("ai.onnx", 11)
    helper.make_operatorsetid("ai.onnx", 18)
]

# producer 主要是和 pytorch 保持一致
model = helper.make_model(graph,
    opset_imports=opset,
    producer_name="pytorch",
    producer_version="2.2.1"
)
onnx.save_model(model, "./../../output/mymodel.onnx")

print(model)
print("All done successfully\n")
