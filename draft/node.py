import numpy as np
from abc import ABCMeta, abstractmethod


class Node(object):
    """docstring for Node"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class TheanoNode(Node):

    def __init__(self):
        pass

    def forward(self):
        pass


class NumpyNode(Node):
    """docstring for NumpyNode"""

    def __init__(self):
        super(NumpyNode, self).__init__()

    def forward(self):
        pass

    def backward(self):
        pass
