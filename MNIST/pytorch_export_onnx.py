#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: pytorch_export_onnx.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2024-10-08.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
@NOTE:
    pip install onnx
    pip install onnxscript
"""

import os
import torch
import torch.onnx
import logging
import logging.config
import networks

if __name__ == "__main__":
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    logging.config.fileConfig("logging.conf")
    inferenceLogger = logging.getLogger("inferenceLog")

    mnist_model = networks.MnistModel(inferenceLogger)
    mnist_network = mnist_model.network.eval()

    # When saving a model to ONNX,
    # PyTorch requires a test batch in proper shape and format.
    # Note that we are picking a BATCH SlZE of 64.
    batch_size = 64
    input_dummy = torch.randn(batch_size, 1, 28, 28).cuda()

    model_path = "./models/"
    os.makedirs(model_path, exist_ok=True)
    model_filename = "model_mnist.onnx"

    # #PyTorch 2.x, NOT commanded on Windows, NOT supported
    # onnx_program = torch.onnx.dynamo_export(mnist_network, input_dummy)
    # #Save the ONNX model in a file
    # onnx_program.save(os.path.join(model_path, model_filename))

    # #based on TorchScript backend and has been available since PyTorch 1.2.0
    torch.onnx.export(
        mnist_network,
        input_dummy,
        os.path.join(model_path, model_filename),
        verbose=True,
    )
