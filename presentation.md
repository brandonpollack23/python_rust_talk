---
title: "Rust 🦀 + Python 🐍"
sub_title: Live Training Visualization
author: Demo Presenter
---

Why Rust + Python?
===

- **Python**: Easy to use, great ML ecosystem
- **Rust**: Blazing fast, memory safe
- **Together**: Best of both worlds!

<!-- pause -->

Use cases:
- Performance-critical libraries
- Real-time visualizations
- Bidirectional communication

<!-- end_slide -->

The Stack
===

```
┌─────────────────────────────┐
│   Python (numpy training)   │
├─────────────────────────────┤
│     PyO3 Bindings Layer     │
├─────────────────────────────┤
│   Rust (ratatui live TUI)   │
└─────────────────────────────┘
```

<!-- pause -->

**Key Technologies:**
- `PyO3` - Rust bindings for Python
- `maturin` - Build tool for PyO3
- `ratatui` - Terminal UI framework
- `numpy` - Training logic

<!-- end_slide -->

The Rust Code
===

```rust
#[pyclass]
struct LiveGraph {
    train_data: Vec<(f64, f64)>,
    val_data: Vec<(f64, f64)>,
    max_epochs: usize,
    title: String,
    terminal: Option<Terminal<...>>,
}

#[pymethods]
impl LiveGraph {
    fn start(&mut self) -> PyResult<()> { ... }
    fn add_point(&mut self, epoch: usize, 
                 train: f64, val: f64) { ... }
    fn draw(&mut self) -> PyResult<bool> { ... }
    fn mark_complete(&mut self) { ... }
    fn stop(&mut self) -> PyResult<()> { ... }
}
```

<!-- end_slide -->

Python Training Logic
===

Python controls the entire training loop:

```python
graph = loss_graph.LiveGraph(100, "Training")
graph.start()
try:
    for epoch in range(100):
        # Your training logic
        train_loss, val_loss = train_step()
        
        # Update visualization
        graph.add_point(epoch, train_loss, val_loss)
        if graph.draw():  # Returns True if 'q' pressed
            break
    graph.mark_complete()
finally:
    graph.stop()  # Always restore terminal
```

<!-- pause -->

No callbacks — simple imperative API!

<!-- end_slide -->

Key Pattern
===

The `try/finally` pattern ensures cleanup:

```python
graph.start()
try:
    # Training loop here
    for epoch in range(max_epochs):
        ...
        graph.add_point(epoch, train, val)
        if graph.draw():
            break
finally:
    graph.stop()  # Terminal always restored!
```

<!-- pause -->

- Python controls the training loop
- Rust only handles terminal rendering
- Clean separation of concerns

<!-- end_slide -->

Live Demo
===

Press `Ctrl+E` to run:

```python +exec
import numpy as np
import loss_graph

np.random.seed(42)
X = np.linspace(-1, 1, 100)
y = X**2 + 0.1 * np.random.randn(100)
weights = np.array([0.0, 0.5, 0.5])

def train_step():
    global weights
    pred = weights[0] + weights[1]*X + weights[2]*X**2
    err = pred - y
    weights -= 0.1 * np.array([
        2*np.mean(err), 2*np.mean(err*X), 2*np.mean(err*X**2)
    ])
    return float(np.mean(err**2)), float(np.mean(err**2) * 1.1)

graph = loss_graph.LiveGraph(50, "Live Demo")
graph.start()
try:
    for epoch in range(50):
        train_loss, val_loss = train_step()
        graph.add_point(epoch, train_loss, val_loss)
        if graph.draw():
            break
    graph.mark_complete()
finally:
    graph.stop()
```

<!-- end_slide -->

Building the Project
===

```bash
# Build and install with uv
uv run maturin develop

# Run!
uv run python demo.py
```

<!-- pause -->

That's it! 🎉

<!-- end_slide -->

Key Takeaways
===

1. **PyO3** creates Python bindings for Rust
2. **Python drives** the training loop — no callbacks
3. **Rust handles** terminal state and rendering
4. **try/finally** ensures clean terminal restoration

<!-- pause -->

Resources:
- PyO3: https://pyo3.rs
- ratatui: https://ratatui.rs
- maturin: https://maturin.rs

<!-- end_slide -->

Thank You!
===

Questions?

<!-- pause -->

```
    🦀 + 🐍 = ❤️
```
