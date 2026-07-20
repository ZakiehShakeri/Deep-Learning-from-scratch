import numpy as np
from .module import Module

class Sigmoid(Module):
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.output * (1 - self.output))

