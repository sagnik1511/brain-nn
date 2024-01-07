"""
This module holds the code for a series / branch of layers called as network
"""


class Sequential:
    """
    Sequential network are series of connected layers
    """

    def __init__(self, layers, **kwargs):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grads):
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
