import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes   # Load the shared library
import matplotlib.pyplot as plt

from nets.ctypes.utils import *

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, batch_shape, stride=None):
        target_stride = stride if stride is not None else self.stride
        return self.custom_conv2d(x, self.weight, batch_shape, self.bias, target_stride, self.padding)


    def custom_conv2d(self, input, weight, batch_shape, bias=None, stride=1, padding=0):
        result = []
        w_max = torch.max(torch.abs(weight)) + 1e-8
        for batch in input:
            result.append(conv1x1(batch, batch_shape, weight / w_max, stride)[0])
        return torch.from_numpy(np.array(result, dtype=np.float32)) * w_max