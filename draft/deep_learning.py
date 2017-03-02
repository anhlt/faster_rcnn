import torch

# Tensor

from torch import Tensor

x = Tensor(5, 3)

x = torch.rand(5, 3)
print(x)

y = torch.rand(5, 3)

print(x + y)

# Numpy Bridge

a = torch.ones(5)
b = a.numpy()
print(a)
print(b)

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
print(b)

# AutoGrad

from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)

y = x + 2

print(y)

z = y * y * 3

out = z.mean()
out.backward()

print(x.grad)
