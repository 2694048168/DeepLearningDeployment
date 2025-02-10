#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: test.py
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
from tqdm import tqdm
import networks

if __name__ == "__main__":
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    logging.config.fileConfig("logging.conf")
    testLogger = logging.getLogger("testLog")

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
    testLogger.info(message_info)
    random.seed(seed)
    torch.manual_seed(seed)

    # Test the PyTorch model
    model_path = "./models/checkpoints/last_ckp.pth"
    checkpoint = torch.load(model_path)
    if "state_dict" in checkpoint.keys():
        checkpoint_weight = checkpoint["state_dict"]
        message_info = f"====> Loading the model from {model_path} successfully"
        testLogger.info(message_info)
    else:
        message_info = f"====> Loading the model from {model_path} NOT successfully"
        testLogger.info(message_info)

    mnist_model = networks.MnistModel(testLogger)
    mnist_model.network.load_state_dict(checkpoint_weight)
    mnist_model.network.eval()

    message_info = f"====> Test the PyTorch model Start"
    testLogger.info(message_info)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        with tqdm(
            total=len(mnist_model.test_loader),
            desc="Iteration inference: ",
            miniters=1,
        ) as tape:
            for data, target in mnist_model.test_loader:
                if torch.cuda.is_available():
                    data = data.to("cuda")
                    target = target.to("cuda")
                output = mnist_model.network(data)
                test_loss += torch.nn.functional.nll_loss(output, target).data.item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()

                tape.set_postfix_str("Batch Loss: %.4f" % test_loss)
                tape.update()

    test_loss /= len(mnist_model.test_loader)
    message_info = (
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(mnist_model.test_loader.dataset),
            100.0 * correct / len(mnist_model.test_loader.dataset),
        )
    )

    testLogger.info(message_info)
    message_info = f"====> Test the PyTorch model End"
    testLogger.info(message_info)
