from abc import ABCMeta, abstractmethod
from numpy.random import uniform
from pprint import pprint
from node import TheanoNode, NumpyNode


class InputSpec(object):
    """
        InputSpec
        Define ndim, dtype and shape of every input to a layer
    """

    def __init__(self, dtype=None, shape=None):
        self.dtype = dtype
        self.shape = shape


class Layer(object):
    """docstring for Layer
        Abstract layer for each particular layer

    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        # low level node process
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, d_out):
        pass

    def call_forward(self, x):
        return self.forward(x)

    def call_backward(self, d_out):
        return self.backward(d_out)

    def out(self, X):
        return self.call_forward(X)

    def computer_gradient(self, d_out):
        self.call_backward(d_out)
        return self.d

    # @abstractmethod
    def build(self, input_shape):
        pass

    # @abstractmethod
    def output_shape(self, input_shape):
        pass

    # @abstractmethod
    def create_input_layer(self):
        pass

    def set_weight(self, weight):
        self.weight = weight

    def get_weight(self):
        return self.weight


class Affine(Layer):

    def __init__(self,
                 input_dim=None,
                 output_dim=None,
                 activation=None,
                 weight=None,
                 init='uniform',
                 **kwargs):

        self.cache = {}
        self.d = {}

        self.init_weight(init)
        super(Affine, self).__init__(**kwargs)

    def init_weight(self, init):
        if type(init == 'str'):
            pass
        self.weight = init((input_dim, output_dim))
        self.b = init((output_dim, ))

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
        return self.d


def init_func(shape):
    return uniform(0, 1, shape)


if __name__ == "__main__":
    input_lenght = 100
    input_dim = 3
    output_dim = 2

    x = uniform(0, 1, (input_lenght, input_dim))

    a = Affine(input_dim=input_dim, output_dim=output_dim, init=init_func)

    result = a.call_forward(x)
    pprint(result.shape)
    pprint(a.cache)

    d_out = uniform(0.1, 0.2, result.shape)

    a.call_backward(d_out)


class ChainNode(object):
    """docstring for ChainNode
        link between nodes in Model.

        layer.add(Node):
            create(ChainNode(self, Node))

    """

    def __init__(self, chain, node):
        self.nodes = chain
        self.node = node

    def add(self, node):
        self.nodes.append(node)

    def next(self):
        location = self.nodes.index(self)
        if not self.end():
            return self.nodes[location + 1]

    def end(self):
        return self.nodes.index(self) + 1 >= len(self.nodes)

    def call(self, messenger):
        r = self.node.call(messenger)

        if r.is_successful() or self.end():
            return r
        else:
            return self.next()(r)


class Layer(object):
    """docstring for Layer"""

    def __init__(self, arg):
        self.arg = arg

    def add(self, node):
        ChainNode(self, node)


