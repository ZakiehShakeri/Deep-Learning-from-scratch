import numpy as np
import sys
from pathlib import Path

# ensure parent directory is on sys.path so we can import the nn package
print(f"Current sys: {str(Path('..').resolve())}")
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from nn.linear import Linear

def numerical_gradient(layer, input, grad_output, epsilon=1e-5):
    grad_W = np.zeros_like(layer.W)
    grad_b = np.zeros_like(layer.b)

    # Compute gradient for weights
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            original_value = layer.W[i, j]
            
            layer.W[i, j] = original_value + epsilon
            loss_plus = np.sum(layer.forward(input) * grad_output)
            
            layer.W[i, j] = original_value - epsilon
            loss_minus = np.sum(layer.forward(input) * grad_output)
            
            grad_W[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            layer.W[i, j] = original_value  # Restore original value

    # Compute gradient for biases
    for i in range(layer.b.shape[0]):
        original_value = layer.b[i]
        
        layer.b[i] = original_value + epsilon
        loss_plus = np.sum(layer.forward(input) * grad_output)
        
        layer.b[i] = original_value - epsilon
        loss_minus = np.sum(layer.forward(input) * grad_output)
        
        grad_b[i] = (loss_plus - loss_minus) / (2 * epsilon)
        layer.b[i] = original_value  # Restore original value

    return grad_W, grad_b

def gradient_check(layer, input, grad_output, epsilon=1e-5, tolerance=1e-7):
    # Compute analytical gradients
    layer.forward(input)
    layer.backward(grad_output)
    analytical_grad_W = layer.grad_W
    analytical_grad_b = layer.grad_b

    # Compute numerical gradients
    numerical_grad_W, numerical_grad_b = numerical_gradient(layer, input, grad_output, epsilon)

    # Check if the gradients are close enough
    weight_close = np.allclose(analytical_grad_W, numerical_grad_W, atol=tolerance)
    bias_close = np.allclose(analytical_grad_b, numerical_grad_b, atol=tolerance)

    return weight_close, bias_close

np.random.seed(42)
input = np.random.randn(5, 3)
grad_output = np.random.randn(5, 2)
layer = Linear(3, 2)
weight_close, bias_close = gradient_check(layer, input, grad_output)
print(f"Weight gradients close: {weight_close}, Bias gradients close: {bias_close}")