from scipy.signal import convolve2d, correlate2d
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Function


class ScipyConv2D(Function):
    """docstring for ScipyConv2D"""

    def forward(input, filter):
        numpy_input = input.numpy()
    # result = correlate2d(input.numpy(), filter.numpy(), mode='valid')     
    result = correlate2d(input.numpy(), filter.numpy(), mode='valid')