from abc import ABCMeta, abstractmethod
from numpy.random import uniform
import numpy as np


class Node(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, value):
        pass

    @abstractmethod
    def backward(self, d_out):
        pass

    @abstractmethod
    def get_params_and_gradient():
        pass


class Affine(Node):

    def __str__(self):
        return 'Affine Node size (%d %d)' % self.shape

    def __init__(self,
                 input_dim=None,
                 output_dim=None,
                 activation=None,
                 weight=None,
                 init='uniform',
                 **kwargs):
        self.shape = (input_dim, output_dim)
        self.cache = {}
        self.d = {}

        self.init_weight(init)
        super(Affine, self).__init__(**kwargs)

    def init_weight(self, init):
        if type(init == 'str'):
            pass
        self.weight = init(self.shape)

        self.b = init((self.shape[1], ))

    def forward(self, x):
        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k)
        and contains a minibatch of N examples,
        where each example x[i] has shape (d_1, ..., d_k).
        We will reshape each input into a vector of dimension
            D = d_1 * ... * d_k
        and then transform it to an output vector of dimension M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        reshaped_x = x.reshape(x.shape[0], -1)
        out = reshaped_x.dot(self.weight) + self.b
        self.cache = {
            'x': x,
            'weight': self.weight,
            'b': self.b
        }
        return out

    def backward(self, d_out):
        x, w = self.cache['x'], self.cache['weight']
        reshaped_x = x.reshape(x.shape[0], -1)

        #######################################################################
        # TODO: Implement the affine backward pass.                           #
        #######################################################################
        dw = reshaped_x.T.dot(d_out)
        dx = d_out.dot(w.T).reshape(*x.shape)
        db = d_out.sum(axis=0)

        self.d = {
            'x': dx,
            'weight': dw,
            'b': db
        }
        return dx

    def get_params_and_gradient(self):
        return {
            'params': self.cache,
            'grads': self.d
        }


class SoftmaxLoss(Node):
    """docstring for Loss"""

    def forward(self, x):
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        self.cache = {
            'x': x,
            'probs': probs
        }
        return probs

    def loss(self, y):
        probs, x = self.cache['probs'], self.cache['x']
        N = x.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        return loss

    def backward(self, y):
        probs, x = self.cache['probs'], self.cache['x']
        N = x.shape[0]
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return dx

    def get_params_and_gradient(self):
        return {}


class ChainNode(object):
    """docstring for ChainNode
        link between nodes in Model.

        layer.add(Node):
            create(ChainNode(self, Node))
    """

    def __init__(self, layer, node):
        self.chains = layer.chains
        self.node = node

    def next(self):
        location = self.chains.index(self)
        print 'locations ', location
        if not self.end():
            return self.chains[location + 1]

    def prev(self):
        location = self.chains.index(self)

        if not self.start():
            return self.chains[location - 1]

    def end(self):
        return self.chains.index(self) + 1 >= len(self.chains)

    def start(self):
        return self.chains.index(self) == 0

    def call_forward(self, input_value):
        r = self.node.forward(input_value)

        if self.end():
            return r
        else:
            return self.next().call_forward(r)

    def call_backward(self, y):
        d_prev = self.node.backward(y)

        if self.start():
            return d_prev
        else:
            return self.prev().call_backward(d_prev)


class Layer(list):
    """docstring for Layer
        Layer is collection of one or many nodes
        It can computer forward and backward
    """

    def __init__(self):
        self.chains = []

    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return self.chains(key)

    def __setitem__(self, key, value):
        self.chains(key, value)

    def add(self, node):
        self.chains.append(ChainNode(self, node))

    def forward(self, value):
        return self.chains[0].call_forward(value)

    def cost_loss(self, value, y):
        self.forward(value)
        loss = self.chains[-1].loss(y)
        return loss

    def backward(self, y):
        return self.chains[-1].call_backward(y)


if __name__ == "__main__":
    input_lenght = 100
    input_dim = 20
    output_dim = 6
    y = np.random.randint(0, 6, (100))
    a = Layer()
    a.add(Affine(input_dim=20, output_dim=15,
                 init=lambda shape: uniform(0, 1, shape)))
    a.add(Affine(input_dim=15, output_dim=10,
                 init=lambda shape: uniform(0, 1, shape)))
    a.add(Affine(input_dim=10, output_dim=5,
                 init=lambda shape: uniform(0, 1, shape)))
    a.add(Affine(input_dim=5, output_dim=6,
                 init=lambda shape: uniform(0, 1, shape)))
    a.add(SoftmaxLoss())

    print a.forward(uniform(1, 2, (input_lenght, input_dim)))
    print a.backward(y).shape
