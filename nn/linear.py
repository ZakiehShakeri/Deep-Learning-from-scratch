import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features)
        self.b = np.random.randn(out_features)
        self.input = None
        self.grad_W = None
        self.grad_b = None
        
    def forward(self, input):
        self.input = input
        return input @ self.W + self.b
    
    def backward(self, grad_output):
        self.grad_W = self.input.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.W.T
        return grad_input
    
    

