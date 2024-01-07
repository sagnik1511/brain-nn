"""
The module holds code for linear / dense layer
"""
import numpy as np

from .common import Neuron


class Linear(Neuron):
    def __init__(self, in_size, out_size, bias=True):
        self.weights = np.random.randn(in_size, out_size)
        self.bias = (
            np.random.randn(
                out_size,
            )
            if bias
            else None
        )

    def forward(self, input_data):
        self.input_data = input_data
        if self.bias is not None:
            return np.dot(self.input_data, self.weights) + self.bias
        return np.dot(self.weights, input_data)

    def backward(self, grads, lr=0.0001):
        d_weights = np.dot(self.input_data.T, grads)
        d_input = np.dot(grads, self.weights.T)
        self.weights -= d_weights * lr
        if self.bias is not None:
            d_bias = np.squeeze(np.sum(grads, axis=0, keepdims=True), axis=0)
            self.bias -= d_bias * lr

        return d_input
