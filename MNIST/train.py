#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: train.py
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
import torch
import random
import argparse
import logging
import logging.config
import networks

if __name__ == "__main__":
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    logging.config.fileConfig("logging.conf")
    rootLogger = logging.getLogger("rootLog")
    trainLogger = logging.getLogger("trainLog")
    """
    var = "Ithaca"
    try:
        int(var)
    except Exception as exp:
        trainLogger.exception(exp)
    """

    # argparse the command-line params
    parser = argparse.ArgumentParser(
        description="Train MNIST network using a PyTorch Models"
    )
    parser.add_argument(
        "-opt", type=str, required=False, help="Path to options JSON file."
    )

    # random seed
    seed = 42
    if seed is None:
        seed = random.randint(1, 10000)
    message_info = f"====> Random Seed: {seed}"
    rootLogger.info(message_info)
    random.seed(seed)
    torch.manual_seed(seed)

    # Train the PyTorch model
    message_info = f"====> Train the PyTorch model Start"
    rootLogger.info(message_info)

    mnist_model = networks.MnistModel(trainLogger)
    mnist_model.learn()

    message_info = f"====> Train the PyTorch model End"
    rootLogger.info(message_info)
