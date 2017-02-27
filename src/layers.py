from abc import ABCMeta, abstractmethod


class InputSpec(object):
    """
        InputSpec
        Define ndim, dtype and shape of every input to a layer
    """

    def __init__(self, dtype=None, shape=None):
        self.dtype = dtype
        self.shape = shape


class Node(object):
    """docstring for Node"""

    def __init__(self,
                 inbound_layers,
                 outbound_layers,
                 input_shape,
                 output_shape):
        super(Node, self).__init__()
        self.inbound_layers = inbound_layers
        self.outbound_layers = outbound_layers
        self.input_shape = input_shape
        self.output_shape = output_shape

    @classmethod
    def create_node(cls,
                    inbound_layers,
                    outbound_layers,
                    input_shape,
                    output_shape):
        return cls(inbound_layers, outbound_layers, input_shape, output_shape)



class Layer(object):
    """docstring for Layer
        Abstract layer for each particular layer

    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        super(Layer, self, kwargs).__init__()
        self.cache = {}
        self.weight = {}
        self.d = {}

    @abstractmethod
    def forward(self, input_value, **parameter):
        return NotImplemented

    @abstractmethod
    def backward(self, d_out):
        return NotImplemented

    def call_forward(self):
        self.forward()

    def call_backward(self):
        self.backward()


class Affine(Layer):
    """docstring for Affine"""

    def __init__(self, arg):
        super(Affine, self).__init__()
