# Deep Learning from Scratch (NumPy)

A minimal deep learning framework implemented from scratch using only NumPy.

This project is part of my journey to refresh and strengthen my AI research skills by reimplementing the core building blocks of modern deep learning without relying on frameworks such as PyTorch or TensorFlow.

The goal is to understand—not just use—the mathematics and software architecture behind neural networks.

---

## Features

Current implementation:

- ✅ Linear Regression
- ✅ Logistic Regression
- ✅ Fully vectorized NumPy implementation
- ✅ Manual gradient computation
- ✅ Gradient Descent optimizer
- ✅ Binary Cross-Entropy Loss
- ✅ Gradient checking using finite differences
- ✅ Linear Layer
- ✅ ReLU
- ✅ Sigmoid
- ✅ Cross-Entropy Loss
- ✅ SGD Optimizer
- ✅ Sequential Model
- ✅ Two-layer Neural Network
- ✅ XOR Classification



Planned:
- ⏳ Softmax
- ⏳ MNIST Digit Classification
- ⏳ Automatic Differentiation
- ⏳ Transformer implementation
- ⏳ PyTorch implementation for comparison

---

## Repository Structure

```
.
├── examples/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   └── xor.py
│
├── nn/
│   ├── linear.py
│   ├── relu.py
│   ├── sigmoid.py
│   ├── losses.py
│   ├── optimizers.py
│   └── sequential.py
│
├── tests/
│   └── gradient_check.py
│
├── notebooks/
│
└── README.md
```

---

## Why build everything from scratch?

Modern frameworks hide many important implementation details.

By implementing every component manually, I aim to gain a deeper understanding of:

- Forward propagation
- Backpropagation
- Computational graphs
- Gradient descent
- Numerical optimization
- Neural network architecture
- Automatic differentiation

---

## Example

```python
layer = Linear(in_features=2, out_features=4)

z = layer.forward(x)

grad_input = layer.backward(grad_output)
```

---

## Mathematical Background

Implemented algorithms are derived directly from first principles.

Examples include:

- Mean Squared Error
- Binary Cross-Entropy
- Sigmoid derivative
- Matrix calculus
- Chain rule
- Numerical gradient checking

---

## Future Roadmap

- [x] Linear Regression
- [x] Logistic Regression
- [x] Gradient Checking
- [x] Linear Layer
- [x] ReLU
- [x] Sigmoid
- [ ] Softmax
- [x] Multi-layer Neural Networks
- [x] XOR
- [ ] MNIST
- [ ] Automatic Differentiation
- [ ] Transformer from Scratch
- [ ] PyTorch Comparison

---

## Requirements

- Python 3.11+
- NumPy
- Matplotlib

Install:

```bash
pip install numpy matplotlib
```

---

## Learning Goals

This repository is intended as a learning project rather than a production deep learning library.

The objective is to understand every component of a modern neural network by implementing it from first principles.

---

## License

MIT