import numpy as np
from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import torch

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


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
