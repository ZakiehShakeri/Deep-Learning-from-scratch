import numpy as np
import sys
from pathlib import Path

print(f"""
This is an example usage of the module for training a linear regression model.
You can use the library by adding the root path of the module to your
`sys.path` like this:
sys.path.insert(0, "{str(Path(__file__).parent.parent.resolve())}")
""")

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from nn.linear import Linear
from nn.losses import MeanSquaredErrorLoss
from nn.optimizers import SGD

# generate random points roughly following y = 2x + 1 with some noise
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1
true_line = 2 * X + 1

model = Linear(1, 1)
loss_fn = MeanSquaredErrorLoss()
optimizer = SGD(learning_rate=0.1)

for epoch in range(100):
    y_pred = model.forward(X)
    loss = loss_fn.forward(y, y_pred)
    grad = loss_fn.backward()
    model.backward(grad)
    optimizer.step(model.parameters(), model.gradients())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
print(f"Final weights: {model.W}, Final bias: {model.b}")
