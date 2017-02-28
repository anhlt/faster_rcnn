from scipy.signal import convolve2d, correlate2d
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch import FloatTensor
import torch
from torch.autograd import Variable


class ScipyConv2DFunction(Function):
    """docstring for ScipyConv2D"""

    def forward(self, input, filter):
        self.save_for_backward(input, filter)
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        return FloatTensor(result)

    def backward(self, grad_output):
        input, filter = self.saved_tensors
        grad_input = convolve2d(grad_output.numpy(),
                                filter.t().numpy(), mode='full')
        grad_filter = convolve2d(
            input.numpy(), grad_output.numpy(), mode='full')
        return FloatTensor(grad_input), FloatTensor(grad_filter)


class ScipyConv2D(Module):
    """docstring for ScipyConv2D"""

    def __init__(self, kh, kw):
        super(ScipyConv2D, self).__init__()
        self.filter = Parameter(torch.randn(kh, kw))

    def forward(self, input):
        print input
        return ScipyConv2DFunction()(input, self.filter)


module = ScipyConv2D(3, 3)

input = Variable(torch.randn(10, 10), requires_grad=True)
output = module(input)

print(output)
output.backward(torch.randn(8, 8))
