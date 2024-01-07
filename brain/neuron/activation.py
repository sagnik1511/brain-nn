"""
This module holds code for activation layers
"""
import numpy as np

from .common import Neuron


class ReLU(Neuron):
    def forward(self, input_data):
        """
        output (y) = x if x > 0
                   = 0 elsewhere
        """
        return np.where(input_data > 0, input_data, 0)

    def backward(self, grads, **kwargs):
        """
        output (y') = 1 if x > 0
                    = 0 elsewhere
        """
        return np.where(grads > 0, 1, 0)


class LeakyReLU(Neuron):
    def __init__(self, slope=0.1):
        self.slope = slope

    def forward(self, input_data):
        """
        output (y) = slope * x if x > 0
                   = 0 elsewhere
        """
        return self.slope * np.where(input_data > 0, input_data, 0)

    def backward(self, grads, **kwargs):
        """
        output (y') = slope if x > 0
                    = 0 elsewhere
        """
        return np.where(grads > 0, self.slope, grads)


class TanH(Neuron):
    def forward(self, input_data):
        return np.tanh(input_data)

    def backward(self, grads):
        return 1 - np.tanh(grads) ** 2


class Sigmoid(Neuron):
    def forward(self, input_data):
        return 1 / (1 + np.exp(-input_data))

    def backward(self, grads):
        return self.forward(grads) * (1 - self.forward(grads))
