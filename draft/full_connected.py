import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

N, D_in, H, D_out = 64, 1000, 100, 10


class AffineFunction(Function):
    """docstring for AffineFunction"""

    def forward(self, input, weight, bias):
        self.save_for_backward(input, weight, bias)
        input = input.numpy()
        weight = weight.numpy()
        bias = bias.numpy()
        reshaped_input = input.reshape(input.shape[0], -1)
        result = reshaped_input.dot(weight) + bias
        return torch.FloatTensor(result)

    def backward(self, grad_output):
        input, weight, bias = self.saved_tensors
        input = input.numpy()
        weight = weight.numpy()
        bias = bias.numpy()
        grad_output = grad_output.numpy()
        reshaped_input = input.reshape(input.shape[0], -1)

        grad_weight = reshaped_input.T.dot(grad_output)
        grad_input = grad_output.dot(weight.T).reshape(*input.shape)
        grad_bias = grad_output.sum(axis=0)

        return torch.FloatTensor(grad_input), torch.FloatTensor(grad_weight), torch.FloatTensor(grad_bias)


class Affine(Module):
    def __init__(self, d_in, d_out):
        super(Affine, self).__init__()
        self.weight = Parameter(torch.randn(d_in, d_out))
        self.bias = Parameter(torch.rand(d_out))

    def forward(self, input):
        return AffineFunction()(input, self.weight, self.bias)


module = Affine(D_in, D_out)
print(list(module.parameters()))

input = Variable(torch.randn(N, D_in), requires_grad=True)

output = module(input)
print(output)

output.backward(torch.randn(N, D_out))
