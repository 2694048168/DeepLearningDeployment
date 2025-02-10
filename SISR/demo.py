#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: demo.py
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

import tensorrt as trt
import numpy as np
import os

import pycuda.driver as cuda
import pycuda.autoinit
import cv2

import matplotlib.pyplot as plt


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:
    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = (
                trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            )
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray, batch_size=2):
        x = x.astype(self.dtype)

        np.copyto(self.inputs[0].host, x.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async(
            batch_size=batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        return [
            out.host.reshape(model.engine.get_binding_shape(1)) for out in self.outputs
        ]


# =============================
if __name__ == "__main__":
    batch_size = 1
    trt_engine_path = os.path.join("models/sr.engine")

    model = TrtModel(trt_engine_path)
    shape = model.engine.get_binding_shape(0)
    shape = model.engine.get_binding_shape(0)

    img_path = "images/butterfly_LRBI_x2.png"

    # 以下是图片预处理
    data = cv2.imread(img_path)
    ori_shape = data.shape[:-1]
    data = cv2.resize(data, (shape[3], shape[2])) / 255.0
    cv2.imshow("1", data)

    data = data.transpose(2, 0, 1)
    data = data[None]

    # 前向传播
    result = model(data, batch_size)[0][0]
    res = np.clip(result, 0, 1)

    sr_img = (res.transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imshow("sr_img", sr_img)
    cv2.waitKey(0)
    cv2.imwrite("images/butterfly_LRBI_x2_SR.png", sr_img)
