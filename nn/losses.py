import numpy as np
from .module import Module

# Binary cross entropy loss 
class BinaryCrossEntropyLoss(Module):
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_true, y_pred):
        """
        Compute the binary cross entropy loss.

        Args:
            y_true (np.ndarray): True labels (0 or 1).
            y_pred (np.ndarray): Predicted probabilities (between 0 and 1).

        Returns:
            float: The binary cross entropy loss.
        """

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        self.y_true = y_true
        self.y_pred = y_pred
        loss = -(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        # average over all samples
        return np.mean(loss)
        

    def backward(self):
        """
        Compute the gradient of the binary cross entropy loss with respect to the predictions.

        Returns:
            np.ndarray: The gradient of the loss with respect to the predictions.
        """
        grad = -(self.y_true / self.y_pred) + ((1 - self.y_true) / (1 - self.y_pred))
        return grad / len(self.y_true)  # Average over all samples

class MeanSquaredErrorLoss(Module):
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_true, y_pred):
        """
        Compute the mean squared error loss.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The mean squared error loss.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        loss = np.mean((self.y_true - self.y_pred) ** 2)
        return loss

    def backward(self):
        """
        Compute the gradient of the mean squared error loss with respect to the predictions.

        Returns:
            np.ndarray: The gradient of the loss with respect to the predictions.
        """
        grad = 2 * (self.y_pred - self.y_true) / len(self.y_true)  # Average over all samples
        return grad