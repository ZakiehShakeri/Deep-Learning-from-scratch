import numpy as np
from matplotlib import pyplot as plt

# generating simple binary classification data
np.random.seed(0)
num_samples = 100
data = np.random.randn(num_samples, 2)
true_labels = (data[:, 0] + data[:, 1] > 0).astype(int)
true_labels = true_labels.reshape(-1, 1)  # Reshape to (num_samples, 1)

data_test = np.random.randn(20, 2)
true_labels_test = (data_test[:, 0] + data_test[:, 1] > 0).astype(int)
true_labels_test = true_labels_test.reshape(-1, 1)

# visualizing dataset
# make sure labels are a 1D array for boolean indexing
labels = true_labels.ravel()
mask0 = labels == 0
mask1 = labels == 1
plt.scatter(data[mask0, 0], data[mask0, 1], color='red', label='Class 0')
plt.scatter(data[mask1, 0], data[mask1, 1], color='blue', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Classification Data')
plt.legend()
plt.show()

# definitions and initializations
def initialize_parameters(n_features):
    w = np.random.randn(n_features, 1)
    b = np.random.randn(1)
    return w, b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w, b = initialize_parameters(2)

# functions definitions
def forward(data, w, b):
    z = np.dot(data, w) + b
    pred_labels = sigmoid(z)
    return pred_labels

# binary cross-entropy loss function
def compute_loss(true_labels, pred_labels):
    return -(true_labels * np.log(pred_labels) + (1 - true_labels) * np.log(1 - pred_labels))

def backward(data, true_labels, pred_labels):
    dl_dz = pred_labels - true_labels
    grad_w = data.T @ dl_dz / len(data)
    grad_b = np.mean(dl_dz)
    return grad_w, grad_b

def update_parameters(learning_rate, w, b, grad_w, grad_b):
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    return w, b

def train(data, true_labels, learning_rate=0.01, num_epochs=1000):
    n_samples, n_features = data.shape
    w, b = initialize_parameters(n_features)
    
    for epoch in range(num_epochs):
        # Forward pass
        pred_labels = forward(data, w, b)
        
        # Compute loss
        loss = compute_loss(true_labels, pred_labels)
        
        # Backward pass (gradient computation)
        grad_w, grad_b = backward(data, true_labels, pred_labels)
        
        # Update parameters
        w, b = update_parameters(learning_rate, w, b, grad_w, grad_b)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {np.mean(loss)}')
    
    print(f'Final Loss: {np.mean(loss)}')
    print(f'Final weights: {w}')
    print(f'Final bias: {b}')
    return w, b

def predict(data_test, w, b):
    pred_probs = forward(data_test, w, b)
    return (pred_probs > 0.5).astype(int)

w, b = train(data, true_labels, learning_rate=0.01, num_epochs=1000)
predicted_labels = predict(data_test, w, b)
accuracy = np.mean(predicted_labels == true_labels_test)
print(f'Accuracy: {accuracy}')
