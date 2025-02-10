#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 06_gen_onnx_plugin.py
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
import torch.onnx
import torch.autograd


class MYSELUImpl(torch.autograd.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # reference:https://pytorch.org/docs/stable/autograd.html
    @staticmethod
    def symbolic(g, x, p):
        print("------------- call symbolic -------------")
        return g.op("MYSELU", x, p,
                    g.op("Constant", value_t=torch.tensor([3, 2, 1], dtype=torch.float32)),
                    attr1_s = "string-attr",
                    attr2_i = [1, 2, 3],
                    attr3_f = 222,
                    )
    
    @staticmethod
    def forward(ctx, x, p):
        return x * 1 / (1 + torch.exp(-x))

class MYSELU(nn.Module):
    def __init__(self, num, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        
        self.param = nn.parameter.Parameter(torch.arange(num).float())

    def forward(self, x):
        return MYSELUImpl.apply(x, self.param)


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        # self.relu = nn.ReLU()
        self.relu = MYSELU(3)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x  = self.conv(x)
        x = self.relu(x)
        return x


# --------------------------
if __name__ == "__main__":
    model = Model().eval()
    input_tensor = torch.tensor([
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [-1, 1, 1],
        [1, 0, 1],
        [1, 1, -1]
    ]
    ], dtype=torch.float32).view(2, 1, 3, 3)

    print(type(input_tensor))

    output = model(input_tensor)
    print(f"The inference result via PyTorch: \n{output}\n")

    dummy = torch.zeros(1, 1, 3, 3)
    torch.onnx.export(model,
                    (dummy,),
                    "./../../output/demo_plugin.onnx",
                    autograd_inlining=False,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                    verbose=True,
                    input_names=["image"],
                    output_names=["output"],
                    # opset_version=11, 
                    opset_version=18, 
                    dynamic_axes={ 
                        "image": {0: "batch", 2: "height", 3: "width"},
                        "output": {0: "batch", 2: "height", 3: "width"}
                    }
    )

    print("All done is successfully\n")
