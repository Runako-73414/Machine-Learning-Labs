import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
import numpy as np
import time

X, y = load_digits(return_X_y=True)

# Scale pixel values
X = X / 16.0

print("Minimum value:", X.min())
print("Maximum value:", X.max())

Y = np.eye(10)[y]

print("Y shape:", Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

# ReLU activation
def relu(x):
    return np.maximum(0, x)

# Softmax activation
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Hidden layer width
W = 64

# Weight initialization
np.random.seed(42)

W1 = np.random.randn(64, W) * 0.01
b1 = np.zeros((1, W))

W2 = np.random.randn(W, W) * 0.01
b2 = np.zeros((1, W))

W3 = np.random.randn(W, 10) * 0.01
b3 = np.zeros((1, 10))

print("Weights initialized successfully!")

# Forward propagation

Z1 = X_train @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = relu(Z2)

Z3 = A2 @ W3 + b3
A3 = softmax(Z3)

print("Output shape:", A3.shape)

# Cross-entropy loss

loss = -np.mean(np.sum(Y_train * np.log(A3 + 1e-8), axis=1))

print("Loss:", loss)

# Calculate accuracy

predictions = np.argmax(A3, axis=1)
actual = np.argmax(Y_train, axis=1)

accuracy = np.mean(predictions == actual)

print("Accuracy:", accuracy)

learning_rate = 0.1

batch_size = 32

losses = []

for epoch in range(1, 51): 

    indices = np.random.permutation(len(X_train))

    X_train = X_train[indices]
    Y_train = Y_train[indices]

    for i in range(0, len(X_train), batch_size):

        X_batch = X_train[i:i + batch_size]
        Y_batch = Y_train[i:i + batch_size]

        # Forward pass
        Z1 = X_batch @ W1 + b1
        A1 = relu(Z1)

        Z2 = A1 @ W2 + b2
        A2 = relu(Z2)

        Z3 = A2 @ W3 + b3
        A3 = softmax(Z3)

        # Loss
        loss = -np.mean(np.sum(Y_batch * np.log(A3 + 1e-8), axis=1))

        # Backpropagation
        dZ3 = (A3 - Y_batch) / len(X_batch)

        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ W3.T
        dZ2 = dA2 * (Z2 > 0)

        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)

        dW1 = X_batch.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Gradient descent
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print("Epoch", epoch, "Loss:", loss)

    losses.append(loss)

# Final training accuracy

Z1 = X_train @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = relu(Z2)

Z3 = A2 @ W3 + b3
A3 = softmax(Z3)

predictions = np.argmax(A3, axis=1)
actual = np.argmax(Y_train, axis=1)

final_accuracy = np.mean(predictions == actual)

print("Final Training Accuracy:", final_accuracy)

# Test accuracy

Z1 = X_test @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = relu(Z2)

Z3 = A2 @ W3 + b3
A3 = softmax(Z3)

test_predictions = np.argmax(A3, axis=1)
test_actual = np.argmax(Y_test, axis=1)

test_accuracy = np.mean(test_predictions == test_actual)

print("Test Accuracy:", test_accuracy)

forward_times = []

for _ in range(100):

    start = time.perf_counter()

    Z1 = X_train @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    end = time.perf_counter()

    forward_times.append(end - start)

median_forward = np.median(forward_times)

print("Median Forward Pass Time:", median_forward)

backward_times = []

for _ in range(100):

    start = time.perf_counter()

    dZ3 = (A3 - Y_train) / len(X_train)

    dW3 = A2.T @ dZ3

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)

    dW2 = A1.T @ dZ2

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X_train.T @ dZ1

    end = time.perf_counter()

    backward_times.append(end - start)

median_backward = np.median(backward_times)

print("Median Backward Pass Time:", median_backward)

epsilon = 1e-5

original_weight = W1[0, 0]

W1[0, 0] = original_weight + epsilon

Z1 = X_train @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = relu(Z2)

Z3 = A2 @ W3 + b3
A3 = softmax(Z3)

loss_plus = -np.mean(
    np.sum(Y_train * np.log(A3 + 1e-8), axis=1)
)

W1[0, 0] = original_weight - epsilon

Z1 = X_train @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = relu(Z2)

Z3 = A2 @ W3 + b3
A3 = softmax(Z3)

loss_minus = -np.mean(
    np.sum(Y_train * np.log(A3 + 1e-8), axis=1)
)

numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)

print("Numerical Gradient:", numerical_gradient)

W1[0, 0] = original_weight

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("lab02/results/loss_curves.png")
plt.close()