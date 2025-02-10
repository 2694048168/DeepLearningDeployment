#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 02_dynamicShape.py
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
import torch.nn.functional as F

weight = torch.FloatTensor([
    [1.0, 2.0, 3.1],
    [0.1, 0.1, 0.1],
    [0.2, 0.2, 0.2],
]).view(1, 1, 3, 3)

bias = torch.FloatTensor([0.0]).view(1)

input = torch.FloatTensor([
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [-1, 1, 1],
        [1, 0, 1],
        [1, 1, -1],
    ]
]).view(2, 1, 3, 3)

print(F.conv2d(input, weight, bias, padding=1))
