#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: infer_tensorrt_onnx.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2024-10-07.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

import os
import logging
import logging.config
import networks
import numpy as np
import torch
import tensorrt as trt
import common


# You can set the logger severity higher to suppress messages
# (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(
        name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE
    )

    def add_matmul_as_fc(net, input, outputs, w, b):
        assert len(input.shape) >= 3
        m = 1 if len(input.shape) == 3 else input.shape[0]
        k = int(np.prod(input.shape) / m)
        assert np.prod(input.shape) == m * k
        n = int(w.size / k)
        assert w.size == n * k
        assert b.size == n

        input_reshape = net.add_shuffle(input)
        input_reshape.reshape_dims = trt.Dims2(m, k)

        filter_const = net.add_constant(trt.Dims2(n, k), w)
        mm = net.add_matrix_multiply(
            input_reshape.get_output(0),
            trt.MatrixOperation.NONE,
            filter_const.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )

        bias_const = net.add_constant(trt.Dims2(1, n), b)
        bias_add = net.add_elementwise(
            mm.get_output(0), bias_const.get_output(0), trt.ElementWiseOperation.SUM
        )

        output_reshape = net.add_shuffle(bias_add.get_output(0))
        output_reshape.reshape_dims = trt.Dims4(m, n, 1, 1)
        return output_reshape

    conv1_w = weights["conv1.weight"].cpu().numpy()
    conv1_b = weights["conv1.bias"].cpu().numpy()
    conv1 = network.add_convolution_nd(
        input=input_tensor,
        num_output_maps=20,
        kernel_shape=(5, 5),
        kernel=conv1_w,
        bias=conv1_b,
    )
    conv1.stride_nd = (1, 1)

    pool1 = network.add_pooling_nd(
        input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2)
    )
    pool1.stride_nd = trt.Dims2(2, 2)

    conv2_w = weights["conv2.weight"].cpu().numpy()
    conv2_b = weights["conv2.bias"].cpu().numpy()
    conv2 = network.add_convolution_nd(
        pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b
    )
    conv2.stride_nd = (1, 1)

    pool2 = network.add_pooling_nd(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride_nd = trt.Dims2(2, 2)

    fc1_w = weights["fc1.weight"].cpu().numpy()
    fc1_b = weights["fc1.bias"].cpu().numpy()
    fc1 = add_matmul_as_fc(network, pool2.get_output(0), 500, fc1_w, fc1_b)

    relu1 = network.add_activation(
        input=fc1.get_output(0), type=trt.ActivationType.RELU
    )

    fc2_w = weights["fc2.weight"].cpu().numpy()
    fc2_b = weights["fc2.bias"].cpu().numpy()
    fc2 = add_matmul_as_fc(
        network, relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b
    )

    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
    # Populate the network using weights from the PyTorch model.
    populate_network(network, weights)
    # Build and return an engine.
    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan)


# Loads a random test case from pytorch's DataLoader
def load_random_test_case(model, pagelocked_buffer):
    # Select an image at random to be the test case.
    img, expected_output = model.get_random_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output


# ---------------------------
# 遵循了用TensorRT推理的基本步骤:
# 1. 参数准备
# 2. 推理引擎构建
# 3. 执行推理
# ---------------------------
if __name__ == "__main__":
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    logging.config.fileConfig("logging.conf")
    inferenceLogger = logging.getLogger("inferenceLog")

    common.add_help(description="Inference MNIST network via PyTorch model")

    model_path = "./models/checkpoints/last_ckp.pth"
    checkpoint = torch.load(model_path)
    if "state_dict" in checkpoint.keys():
        checkpoint_weight = checkpoint["state_dict"]
        message_info = f"====> Loading the model from {model_path} successfully"
        inferenceLogger.info(message_info)
    else:
        message_info = f"====> Loading the model from {model_path} NOT successfully"
        inferenceLogger.info(message_info)

    mnist_model = networks.MnistModel(inferenceLogger)
    weights = checkpoint_weight

    # Do inference with TensorRT.
    engine = build_engine(weights)
    message_info = f"====> Build the engine to inference via TensorRT successfully"
    inferenceLogger.info(message_info)

    # Build an engine, allocate buffers and create a stream.
    # For more information on buffer allocation, refer to the introductory samples.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
    # For more information on performing inference, refer to the introductory samples.
    # The common.do_inference function will return a list of outputs - we only have one in this case.
    [output] = common.do_inference(
        context,
        engine=engine,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    pred = np.argmax(output)
    common.free_buffers(inputs, outputs, stream)

    message_info = f"Test Case: {str(case_num)}"
    inferenceLogger.info(message_info)
    message_info = f"Prediction: {str(pred)}"
    inferenceLogger.info(message_info)
