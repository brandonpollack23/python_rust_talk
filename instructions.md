# Rust + Python Live Training Demo

A demonstration of using Rust with ratatui to create a real-time training visualization, where Python performs the training logic and Rust handles the TUI rendering.

## Prerequisites

- Rust (1.70+)
- Python (3.8+)
- uv (https://docs.astral.sh/uv/)
- maturin (`uv tool install maturin` or `uv add maturin`)
- presenterm (`cargo install presenterm`)

## Quick Start

### 1. Build and install the Python module

```bash
# Build and install in development mode using uv
uv run maturin develop --uv
```

### 2. Run the demo

```bash
uv run python demo.py
```

Watch as Python performs gradient descent and Rust visualizes the loss in real-time. Press `q` to exit.

### 3. Run the presentation

```bash
presenterm presentation.md
```

Use arrow keys to navigate. On slides with executable code, press `Ctrl+E` to run the code.

## Project Structure

```
.
├── Cargo.toml          # Rust dependencies
├── pyproject.toml      # Python build config (includes numpy)
├── src/
│   └── lib.rs          # Rust LiveGraph class with PyO3 bindings
├── demo.py             # Python training simulation with numpy
├── presentation.md     # presenterm slides
├── explanation.md      # Detailed architecture explanation
└── instructions.md     # This file
```

## How It Works

1. **Python Training Logic** (`demo.py`): Implements polynomial regression with gradient descent using numpy. The `train_step(epoch)` function performs one SGD update and returns `(train_loss, val_loss)`.

2. **Rust Visualization** (`src/lib.rs`): The `LiveGraph` class:
   - Accepts a Python callable in its `run()` method
   - Calls it each frame to get new loss values
   - Updates the chart in real-time
   - Handles keyboard input for quitting

3. **Bidirectional Communication**: Rust controls the render loop but Python controls the training logic. Each frame:
   - Rust calls `train_step_fn(epoch)` into Python
   - Python computes gradients, updates weights, returns losses
   - Rust extracts the tuple and updates the visualization

## API Reference

```python
import loss_graph
import numpy as np

# Create a live graph
graph = loss_graph.LiveGraph(
    max_epochs=100,    # Total epochs to run
    title="My Training"  # Graph title
)

# Define your training step
def train_step(epoch: int) -> tuple[float, float]:
    # Your training logic here (numpy, PyTorch, etc.)
    # ...
    return (train_loss, val_loss)

# Run the visualization
graph.run(train_step)
```

## Extending for PyTorch

```python
import torch
import loss_graph

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

def train_step(epoch: int) -> tuple[float, float]:
    optimizer.zero_grad()
    loss = compute_loss(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        val_loss = compute_loss(model(X_val), y_val)
    
    return (loss.item(), val_loss.item())

graph = loss_graph.LiveGraph(100, "PyTorch Training")
graph.run(train_step)
```
