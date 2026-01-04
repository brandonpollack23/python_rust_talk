#!/usr/bin/env python3
"""Demo script showing live training visualization with Rust TUI."""

from numpy import floating
from numpy.typing import NDArray
from typing import Any

import numpy as np
import loss_graph

print("🚀 Launching Rust-powered TUI with live Python training...")
print("   Simulating gradient descent optimization")
print("   Press 'q' to exit the visualization\n")

# Simulate a simple neural network training scenario
# We're "learning" to fit a quadratic function
np.random.seed(42)

# Generate some fake "training data"
X_train: NDArray = np.linspace(-1, 1, 100)
y_train: NDArray = X_train**2 + 0.1 * np.random.randn(100)
X_val: NDArray = np.linspace(-1, 1, 20)
y_val: NDArray = X_val**2 + 0.1 * np.random.randn(20)

# Simple model: y = w2*x^2 + w1*x + w0
# Start with random weights
weights: NDArray = np.random.rand(3)
learning_rate = 0.1


def compute_loss(X: NDArray, y: NDArray, w: NDArray) -> floating[Any]:
    """MSE loss for polynomial regression."""
    pred = w[0] + w[1] * X + w[2] * X**2
    return np.mean((pred - y) ** 2)


def train_step(epoch: int) -> tuple[float, float]:
    """Perform one training step and return (train_loss, val_loss)."""
    global weights

    # Compute gradients (analytical for this simple case)
    pred = weights[0] + weights[1] * X_train + weights[2] * X_train**2
    error = pred - y_train

    grad_w0 = 2 * np.mean(error)
    grad_w1 = 2 * np.mean(error * X_train)
    grad_w2 = 2 * np.mean(error * X_train**2)

    # SGD update with some noise to simulate mini-batch
    noise = 0.02 * np.random.randn(3) * max(0.1, 1 - epoch / 100)
    weights[0] -= learning_rate * grad_w0 + noise[0]
    weights[1] -= learning_rate * grad_w1 + noise[1]
    weights[2] -= learning_rate * grad_w2 + noise[2]

    train_loss = compute_loss(X_train, y_train, weights)
    val_loss = compute_loss(X_val, y_val, weights)

    return (float(train_loss), float(val_loss))


# Create live graph and run training loop from Python
max_epochs = 100
graph = loss_graph.LiveGraph(
    max_epochs=max_epochs, title="Polynomial Regression Training"
)

graph.start()
try:
    for epoch in range(max_epochs):
        train_loss, val_loss = train_step(epoch)
        graph.add_point(float(epoch), train_loss, val_loss)

        if graph.draw():  # Returns True if user pressed 'q'
            break
    else:
        # Training completed, show final state
        graph.mark_complete()
        while not graph.draw():
            pass  # Wait for user to press 'q'
finally:
    graph.stop()

print("\n✅ Training complete!")
print(
    f"   Final weights: w0={weights[0]:.4f}, w1={weights[1]:.4f}, w2={weights[2]:.4f}"
)
print("   (Target: w0≈0, w1≈0, w2≈1)")
