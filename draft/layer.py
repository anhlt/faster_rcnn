import theano
from numpy import random
from matplotlib import pyplot as plt
import numpy as np
from theano import tensor as T


class LogicticRegression(object):
    """docstring for LogicticRegression"""

    def __init__(self, input, n_in, n_out):
        super(LogicticRegression, self).__init__()
        self.input = input
        self.n_in = n_in
        self.n_out = n_out

        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros((n_out, ), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startwith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
