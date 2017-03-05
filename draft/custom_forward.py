import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module


class AffineFunction(Function):
    """docstring for AffineFunction"""

    def forward(self, input):
    	input = input.numpy()

    def backward(self, grad_output):
        pass


class Affine(Module):
    """docstring for Affine"""

    def __init__(self):
        super(Affine, self).__init__()

    def forward(self, input):
        return AffineFunction()(input)


module = Affine()
