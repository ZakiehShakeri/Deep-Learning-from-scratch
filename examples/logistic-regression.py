import sys
import numpy as np
from pathlib import Path

print(f"""
This is an example usage of the module for training a logistic regression model.
You can use the library by adding the root path of the module to your
`sys.path` like this:
sys.path.insert(0, "{str(Path(__file__).parent.parent.resolve())}")
""")

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from nn.linear import Linear
from nn.optimizers import SGD
from nn.sequential import Sequential
from nn.sigmoid import Sigmoid
from nn.losses import BinaryCrossEntropyLoss as BCE

# generating simple binary classification data
np.random.seed(0)
num_samples = 100
data = np.random.randn(num_samples, 2)
true_labels = (data[:, 0] + data[:, 1] > 0).astype(int)
true_labels = true_labels.reshape(-1, 1)  # Reshape to (num_samples, 1)

data_test = np.random.randn(20, 2)
true_labels_test = (data_test[:, 0] + data_test[:, 1] > 0).astype(int)
true_labels_test = true_labels_test.reshape(-1, 1)


model = Sequential(
    Linear(2, 1),
    Sigmoid()
)
loss_fn = BCE()
optimizer = SGD(learning_rate=0.01)

def train(model, data, true_labels, loss_fn, optimizer, epochs=1000):
    for epoch in range(epochs):
        # Forward pass
        pred_labels = model.forward(data)

        # Compute loss
        loss = loss_fn.forward(true_labels, pred_labels)

        # Backward pass (gradient computation)
        grad = loss_fn.backward()
        model.backward(grad)

        # Update parameters
        optimizer.step(model.parameters(), model.gradients())
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {np.mean(loss)}')

    print(f'Final Loss: {np.mean(loss)}')
    print(f'Final weights: {model.layers[0].W}')
    print(f'Final bias: {model.layers[0].b}')

def test(model, data_test, true_labels_test):
    pred_labels_test = model.forward(data_test)
    pred_labels_test = (pred_labels_test > 0.5).astype(int)
    accuracy = np.mean(pred_labels_test == true_labels_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

train(model, data, true_labels, loss_fn, optimizer, epochs=1000)
test(model, data_test, true_labels_test)