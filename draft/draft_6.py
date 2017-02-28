import torch
from torch.autograd import Variable
from torch.autograd import Function
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class ScipyConv2dFunction(Function):
    """docstring for ScipyConv2dFunction"""

    def forward(self, input, filter):
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        self.save_for_backward(input, filter)
        return torch.FloatTensor(result)

    def backward(self, grad_output):
        input, filter = self.saved_tensors
        grad_input = convolve2d(grad_output.numpy(),
                                filter.t().numpy(), mode='full')
        grad_filter = convolve2d(
            input.numpy(), grad_output.numpy(), mode='valid')
        return torch.FloatTensor(grad_input), torch.FloatTensor(grad_filter)


class ScipyConv2d(Module):

    def __init__(self, kh, kw):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(kh, kw))

    def forward(self, input):
        return ScipyConv2dFunction()(input, self.filter)


module = ScipyConv2d(3, 3)

print(list(module.parameters()))
input = Variable(torch.randn(10, 10), requires_grad=True)
output = module(input)
print(output)

output.backward(torch.randn(8, 8))

print(input.grad)
