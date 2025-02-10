#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: networks.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2024-10-06.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

import os
import torch
import torchvision
import numpy as np
import random


# Network for MNIST
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = torch.nn.Linear(800, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        # x ---> [batch_size, 1, 28, 28]
        x = torch.nn.functional.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = torch.nn.functional.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = x.view(-1, 800)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


# Model for MNIST
class MnistModel(object):
    def __init__(self, trainLogger):
        self.logger = trainLogger
        self.batch_size = 64
        self.test_batch_size = 100
        self.learning_rate = 0.0025
        self.sgd_momentum = 0.9

        self.checkpoint_dir = "./models/checkpoints/"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_ckp_step = 100
        self.save_ckp_epoch = 1
        self.best_epoch = 0
        self.model_dir = "./models/"
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_interval = 100

        # Fetch MNIST data set
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "./data/mnist",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600,
        )

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "./data/mnist",
                train=False,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            ),
            batch_size=self.test_batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600,
        )

        self.network = Network()
        if torch.cuda.is_available():
            self.network = self.network.to("cuda")

        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.learning_rate,
            momentum=self.sgd_momentum,
        )

    # Train the network for one or more epochs, validating after each epoch.
    def learn(self, num_epochs=2):
        # Train the network for a single epoch
        def train(epoch):
            self.network.train()
            # Epoch, Batch size, Iteration
            for batch, (data, target) in enumerate(self.train_loader):
                message_info = f"input tensor data shape: {data.shape}"
                self.logger.info(message_info)
                if torch.cuda.is_available():
                    data = data.to("cuda")
                    target = target.to("cuda")
                data, target = torch.autograd.Variable(data), torch.autograd.Variable(
                    target
                )
                self.optimizer.zero_grad()
                output = self.network(data)
                loss = torch.nn.functional.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                if batch % self.log_interval == 0:
                    message_info = f"Train Epoch: {epoch} [{batch * len(data)}/{len(self.train_loader.dataset)} ({100.0 * batch / len(self.train_loader):.0f}%)]\tLoss: {loss.data.item():.6f}"
                    self.logger.info(message_info)
                self.save_checkpoint(batch)

            # record the best epoch
            self.best_epoch = epoch
            self.save_checkpoint(epoch, is_epoch=True)

        # Test the network
        def test(epoch):
            self.network.eval()
            test_loss = 0
            correct = 0
            for data, target in self.test_loader:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        data = data.to("cuda")
                        target = target.to("cuda")
                    data, target = torch.autograd.Variable(
                        data
                    ), torch.autograd.Variable(target)
                output = self.network(data)
                test_loss += torch.nn.functional.nll_loss(output, target).data.item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()
            test_loss /= len(self.test_loader)
            message_info = (
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(self.test_loader.dataset),
                    100.0 * correct / len(self.test_loader.dataset),
                )
            )
            self.logger.info(message_info)

        # training and loss-test
        for epoch in range(num_epochs):
            train(epoch + 1)
            test(epoch + 1)

    # the network weights, not the training params
    def get_weights(self):
        return self.network.state_dict()

    def save_checkpoint(self, interval, is_epoch=False, is_best=False):
        filename = os.path.join(self.checkpoint_dir, "last_ckp.pth")
        message_info = f"====> Saving the checkpoint to {filename}"
        self.logger.info(message_info)

        if is_epoch:
            ckp = {
                "epoch": interval,
                "state_dict": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_epoch": self.best_epoch,
            }
        else:
            ckp = {
                "iteration": interval,
                "state_dict": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_epoch": self.best_epoch,
            }

        torch.save(ckp, filename)
        if is_epoch:
            if interval % self.save_ckp_epoch == 0:
                message_info = f"====> Saving checkpoint {iter} to {filename.replace('last_ckp','epoch_%d_ckp'%interval)}"
                self.logger.info(message_info)
                torch.save(ckp, filename.replace("last_ckp", "epoch_%d_ckp" % interval))
        else:
            if interval % self.save_ckp_step == 0:
                message_info = f"====> Saving checkpoint {iter} to {filename.replace('last_ckp','iteration_%d_ckp'%interval)}"
                self.logger.info(message_info)
                torch.save(
                    ckp, filename.replace("last_ckp", "iteration_%d_ckp" % interval)
                )

        if is_best:
            message_info = f"====> Saving best checkpoint to {filename.replace('last_ckp','best_ckp')}"
            self.logger.info(message_info)
            torch.save(ckp, filename.replace("last_ckp", "best_ckp"))

    def get_random_testcase(self):
        data, target = next(iter(self.test_loader))
        case_num = random.randint(0, len(data) - 1)
        test_case = data.cpu().numpy()[case_num].ravel().astype(np.float32)
        test_name = target.cpu().numpy()[case_num]
        return test_case, test_name
