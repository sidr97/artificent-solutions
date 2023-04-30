"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(8)  # DO NOT MODIFY!
np.random.seed(8)  # DO NOT MODIFY!


# write your code here ...

def process_block(x, conv_layer, bn_layer, sigmoid_layer):
    # apply multiplication of (conv) and (conv+sigmoid)
    return bn_layer(conv_layer(x)) * sigmoid_layer(bn_layer(conv_layer(x)))


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.Conv_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_0 = nn.BatchNorm2d(32)
        self.Sigmoid_0 = nn.Sigmoid()

        self.Conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.Sigmoid_1 = nn.Sigmoid()

        self.Conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.Sigmoid_2 = nn.Sigmoid()

        self.Conv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.Sigmoid_3 = nn.Sigmoid()

        self.Conv_4 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.Sigmoid_4 = nn.Sigmoid()

        self.Conv_5 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.Sigmoid_5 = nn.Sigmoid()

        self.Conv_6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_6 = nn.BatchNorm2d(64)
        self.Sigmoid_6 = nn.Sigmoid()

        self.Conv_7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_7 = nn.BatchNorm2d(64)
        self.Sigmoid_7 = nn.Sigmoid()

        self.Conv_8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_8 = nn.BatchNorm2d(64)
        self.Sigmoid_8 = nn.Sigmoid()

        self.Conv_9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_9 = nn.BatchNorm2d(64)
        self.Sigmoid_9 = nn.Sigmoid()

        self.Conv_10 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn_10 = nn.BatchNorm2d(256)
        self.Sigmoid_10 = nn.Sigmoid()

        self.MaxPool_0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.Conv_11 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.bn_11 = nn.BatchNorm2d(128)
        self.Sigmoid_11 = nn.Sigmoid()

        self.Conv_12 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.bn_12 = nn.BatchNorm2d(128)
        self.Sigmoid_12 = nn.Sigmoid()

        self.Conv_13 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn_13 = nn.BatchNorm2d(128)
        self.Sigmoid_13 = nn.Sigmoid()

        self.Gemm_0 = nn.Linear(256, 256)
        self.bn_14 = nn.BatchNorm2d(256)
        self.Sigmoid_14 = nn.Sigmoid()

        self.apply(self._init_weights)  # initialising weights as mentioned in the question

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        op = process_block(x, self.Conv_0, self.bn_0, self.Sigmoid_0)

        op = process_block(op, self.Conv_1, self.bn_1, self.Sigmoid_1)

        op = process_block(op, self.Conv_2, self.bn_2, self.Sigmoid_2)

        op = process_block(op, self.Conv_3, self.bn_3, self.Sigmoid_3)

        branch_left_op = process_block(op, self.Conv_4, self.bn_4, self.Sigmoid_4)

        branch_right_op_layer5 = process_block(op, self.Conv_5, self.bn_5, self.Sigmoid_5)
        branch_right_op_layer6 = process_block(branch_right_op_layer5, self.bn_6, self.Conv_6, self.Sigmoid_6)
        branch_right_op_layer7 = process_block(branch_right_op_layer6, self.bn_7, self.Conv_7, self.Sigmoid_7)
        branch_right_op_layer8 = process_block(branch_right_op_layer7, self.bn_8, self.Conv_8, self.Sigmoid_8)
        branch_right_op_layer9 = process_block(branch_right_op_layer8, self.bn_9, self.Conv_9, self.Sigmoid_9)

        op = torch.cat([branch_left_op,
                        branch_right_op_layer5,
                        branch_right_op_layer7,
                        branch_right_op_layer9], dim=1)

        op = process_block(op, self.Conv_10, self.bn_10, self.Sigmoid_10)
        maxpool_op = self.MaxPool_0(op)

        branch_left_op = process_block(maxpool_op, self.Conv_11, self.bn_11, self.Sigmoid_11)

        branch_right_op_layer12 = process_block(op, self.Conv_12, self.bn_12, self.Sigmoid_12)
        branch_right_op_layer13 = process_block(branch_right_op_layer12, self.Conv_13, self.bn_13, self.Sigmoid_13)

        op = torch.cat([branch_left_op, branch_right_op_layer13], dim=1)

        op = op.permute(0, 2, 3, 1)

        op = self.Gemm_0(op)

        op = op.permute(0, 3, 1, 2)
        op = self.bn_14(op)

        op = self.Sigmoid_14(op)

        return op


sample_input = torch.randn((1, 3, 160, 320))

model = Model()

output = model(sample_input)
