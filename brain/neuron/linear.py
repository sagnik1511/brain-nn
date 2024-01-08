"""
The module holds code for linear / dense layer
"""
import numpy as np

from .common import Neuron


class Linear(Neuron):
    def __init__(self, in_size, out_size, bias=True):
        self.weights = np.random.randn(in_size, out_size)
        self.bias = np.zeros((1, out_size)) if bias else None

    def forward(self, input_data):
        self.input_data = input_data
        return (
            np.dot(input_data, self.weights) + self.bias
            if self.bias is not None
            else np.dot(input_data, self.weights)
        )

    def backward(self, grads, lr=0.001):
        d_input = np.dot(grads, self.weights.T)
        d_weights = np.dot(self.input_data.T, grads)
        d_bias = np.sum(self.bias, axis=0)
        self.weights -= d_weights * lr
        self.bias -= d_bias * lr
        return d_input
