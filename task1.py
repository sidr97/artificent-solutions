"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(8)  # DO NOT MODIFY!
np.random.seed(8)  # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)


# now write your custom layer
class CustomGroupedConv2D(nn.Module):
    def __init__(self,
                 w_given,
                 b_given,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super().__init__()
        self.kernel_list = [nn.Conv2d(in_channels=in_channels // groups,  # 4
                                      out_channels=out_channels // groups,  # 8
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=1,
                                      bias=bias,
                                      padding_mode=padding_mode,
                                      device=device,
                                      dtype=dtype) for _ in range(groups)]

        self.out_per_group = out_channels // groups  # num params per group for out_channels
        self.in_per_group = in_channels // groups  # num params per group for in_channels
        with torch.no_grad():
            for index in range(len(self.kernel_list)):
                self.kernel_list[index].weight.copy_(
                    w_given[index * self.out_per_group:(index + 1) * self.out_per_group, :, :, :]
                )
                self.kernel_list[index].bias.copy_(
                    b_given[index * self.out_per_group: (index + 1) * self.out_per_group])

    def forward(self, x):
        res = []
        for k_idx in range(len(self.kernel_list)):
            res.append(self.kernel_list[k_idx](
                x[:, k_idx * self.in_per_group:(k_idx + 1) * self.in_per_group, :, :]))
        return torch.cat(res, dim=1)


# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
custom_layer = CustomGroupedConv2D(w_given=w_torch,
                                   b_given=b_torch,
                                   in_channels=64,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=16,
                                   bias=True)

z = custom_layer(x)

# due to numerical differences, outputs of Conv2D and CustomGroupedConv2D can be slightly different
# resulting in print(y==z) coming out to be false
# Hence using torch.isclose to compare
print(torch.isclose(y, z))
