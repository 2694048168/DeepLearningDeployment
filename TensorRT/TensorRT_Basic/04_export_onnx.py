#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 04_export_onnx.py
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


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=True)
        self.conv.weight.data.fill_(0.3)
        self.conv.bias.data.fill_(0.2)

    def forward(self, x):
        x = self.conv(x)
        # return x.view(x.size(0), -1) #export_error.onnx
        return x.view(-1, int(x.numel() // x.size(0))) #export_right.onnx

# ---------------------------
if __name__ == "__main__":
    model = Model().eval()
    x = torch.full((1, 1, 3, 3), 1.0)
    y = model(x)

    # Pytorch 2.x 进一步优化了 onnx 的导出, 是否两则是一致的, 没有任何区别
    # 推荐所有工具和库, 尽可能使用最新的版本, 毕竟很多工程师进行优化的

    # torch.onnx.export(model, (x, ), "./../../output/export_error.onnx", verbose=True)
    torch.onnx.export(model, (x, ), "./../../output/export_right.onnx", verbose=True)
    print("All done is successfully\n")
