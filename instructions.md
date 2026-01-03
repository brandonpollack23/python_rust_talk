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
# Build and install in development mode
uv sync
uv run maturin develop --release
```

### 2. Run the demo

```bash
# Use --no-sync to prevent uv from rebuilding the package
uv run --no-sync python demo.py
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

1. **Python Training Logic** (`demo.py`): Implements polynomial regression with gradient descent using numpy. Python controls the entire training loop.

2. **Rust Visualization** (`src/lib.rs`): The `LiveGraph` class provides:
   - `start()` / `stop()` — enter/exit TUI mode
   - `add_point()` — add a data point
   - `draw()` — render and check for quit
   - `mark_complete()` — update status display

3. **Python-Driven Architecture**: Python controls everything:
   - Calls `graph.start()` to enter TUI mode
   - Runs the training loop with `for epoch in range(...)`
   - Calls `graph.add_point()` and `graph.draw()` each iteration
   - Uses `try/finally` to ensure `graph.stop()` is called for cleanup

## API Reference

```python
import loss_graph

# Create a live graph
graph = loss_graph.LiveGraph(
    max_epochs=100,    # Total epochs (for X-axis scaling)
    title="My Training"  # Graph title
)

# Enter TUI mode
graph.start()

try:
    for epoch in range(100):
        # Your training logic here
        train_loss, val_loss = train_step()
        
        # Add data point
        graph.add_point(epoch, train_loss, val_loss)
        
        # Render and check for quit ('q' key)
        if graph.draw():
            break
    
    # Mark training as complete (updates title)
    graph.mark_complete()
finally:
    # Always restore terminal state
    graph.stop()
```

### Methods

| Method | Description |
|--------|-------------|
| `LiveGraph(max_epochs, title)` | Create a new graph instance |
| `start()` | Enter TUI mode (raw mode, alternate screen) |
| `add_point(epoch, train_loss, val_loss)` | Add a data point to the graph |
| `draw()` | Render the chart; returns `True` if 'q' pressed |
| `mark_complete()` | Mark training as complete (updates title) |
| `stop()` | Exit TUI mode and restore terminal |

## Extending for PyTorch

```python
import torch
import loss_graph

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
graph = loss_graph.LiveGraph(100, "PyTorch Training")

graph.start()
try:
    for epoch in range(100):
        optimizer.zero_grad()
        loss = compute_loss(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            val_loss = compute_loss(model(X_val), y_val)
        
        graph.add_point(epoch, loss.item(), val_loss.item())
        if graph.draw():
            break
    graph.mark_complete()
finally:
    graph.stop()
```
