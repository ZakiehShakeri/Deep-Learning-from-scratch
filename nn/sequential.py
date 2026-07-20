from .module import Module

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def gradients(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.gradients())
        return grads