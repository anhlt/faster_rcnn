import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module


class AffineFunction(Function):
    """docstring for AffineFunction"""
    def forward(self, input, weight, bias):
        input = input.numpy()
        weight = weight.numpy()
        bias = bias.numpy()
        reshaped_input= input.reshape(input.shape[0], -1)
        result = reshaped_input.dot(weight) + bias



