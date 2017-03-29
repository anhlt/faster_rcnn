import numpy as np
from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import torch


class SoftMaxFunction(Function):
    """docstring for SoftMaxFunction"""

    def forward(self, input, ):
        input = input.numpy()
        probs = np.exp(input - np.max(input, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return torch.FloatTensor(probs)

    def backward(self, grad_output):
        probs = self.saved_tensors
        probs = probs.numpy()
        grad_output = grad_output.numpy()

        d_input = probs.copy()
        d_input[np.arange(N), ]
