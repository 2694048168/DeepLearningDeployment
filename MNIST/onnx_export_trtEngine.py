#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: onnx_export_trtEngine.py
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
"""

import os
import subprocess

# ===========================
if __name__ == "__main__":
    model_path = "./models/"
    os.makedirs(model_path, exist_ok=True)
    model_filename = "model_mnist_Engine.trt"

    onnx_filepath = "./models/model_mnist.onnx"
    engine_filepath = os.path.join(model_path, model_filename)

    USE_FP16 = True

    if USE_FP16:
        command_args = [
            "trtexec",
            f"--onnx={onnx_filepath}",
            f"--saveEngine={engine_filepath}",
            "--inputIOFormats=fp16:chw",
            "--outputIOFormats=fp16:chw",
            "--fp16",
        ]
    else:
        command_args = [
            "trtexec",
            f"--onnx={onnx_filepath}",
            f"--saveEngine={engine_filepath}",
        ]

    # -------------------------------------------------------------
    # https://docs.python.org/zh-cn/3.12/library/subprocess.html
    result = subprocess.run(
        command_args,
        capture_output=True,  # comment this line will output INFO to console
        encoding="utf-8",
        shell=True,
        check=True,
    )

    messageInfo_str = result.stdout
    messageError_str = result.stderr
    returnCode = result.check_returncode()

    logs_path = "./logs/"
    os.makedirs(logs_path, exist_ok=True)
    log_filepath = os.path.join(logs_path, "onnx_convert_engine.log")
    with open(log_filepath, "w") as file_out:
        if messageInfo_str:
            file_out.write(messageInfo_str)
        if messageError_str:
            file_out.write(messageError_str)
        if not returnCode:
            file_out.write("\n\nThe command execute successfully\n")

    # print(f"The command return status code: {returnCode}")
    if not returnCode:
        print("\nThe command execute successfully\n")
    else:
        print("\nThe command execute NOT successfully\n")
