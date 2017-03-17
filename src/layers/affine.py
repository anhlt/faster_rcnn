import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import numpy as np
from torch.nn.parameter import Parameter


class AffineFunction(Function):

    def forward(self, input, weight, bias):
        """
            Computes the forward pass for an affine (fully-connected) layer.
        """
        self.save_for_backward(input, weight, bias)
        input = input.numpy()
        weight = weight.numpy()
        bias = bias.numpy()
        out = input.reshape(input.shape[0], -1).dot(weight) + bias
        return torch.FloatTensor(out)

    def backward(self, grad_output):
        """
            Computes the backward pass for an affine layer.
        """
        input, weight, bias = self.saved_tensors
        input = input.numpy()
        weight = weight.numpy()
        bias = bias.numpy()
        grad_output = grad_output.numpy()
        grad_input = grad_output.dot(weight.T).reshape(input.shape)
        grad_weight = input.reshape(input.shape[0], -1).T.dot(grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        return torch.FloatTensor(grad_input), torch.FloatTensor(grad_weight), torch.FloatTensor(grad_bias)


class Affine(Module):
    """docstring for Affine"""

    def __init__(self, n_in, n_out):
        super(Affine, self).__init__()
        self.weight = Parameter(torch.from_numpy(
            np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)).astype(np.float32)
        ))

        self.bias = Parameter(torch.from_numpy(
            np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_out, )).astype(np.float32)
        ))

    def forward(self, input):
        return AffineFunction()(input, self.weight, self.bias)
