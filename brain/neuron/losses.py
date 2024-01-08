"""
This module holds loss functions
"""
import numpy as np

from brain.neuron.common import Neuron


class MSE(Neuron):
    def forward(y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def backward(y_pred, y_true):
        return 2 * (y_pred - y_true) / len(y_true)


class BCE(Neuron):
    def forward(self, y_pred, y_true):
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(
            y_pred, epsilon, 1 - epsilon
        )  # Clip values to avoid numerical instability
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)

    def backward(self, y_pred, y_true):
        epsilon = 1e-15  # Small constant to avoid division by zero

        y_pred = np.clip(
            y_pred, epsilon, 1 - epsilon
        )  # Clip values to avoid numerical instability

        # Compute the gradient
        gradient = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

        return gradient / len(y_true)
