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
    train_data: Arc<Mutex<Vec<(f64, f64)>>>,
    val_data: Arc<Mutex<Vec<(f64, f64)>>>,
    max_epochs: usize,
    title: String,
}

#[pymethods]
impl LiveGraph {
    fn run(&self, py: Python<'_>, 
           train_step_fn: PyObject) -> PyResult<()> {
        // ... render loop calls Python each frame
        let result = train_step_fn.call1(py, (epoch,))?;
        let (train_loss, val_loss): (f64, f64) = 
            result.extract(py)?;
        self.add_point(epoch, train_loss, val_loss);
    }
}
```

<!-- end_slide -->

Python Training Logic
===

Real gradient descent with numpy:

```python
def train_step(epoch: int) -> tuple[float, float]:
    global weights
    
    # Compute gradients
    pred = weights[0] + weights[1]*X + weights[2]*X**2
    error = pred - y_train
    
    # SGD update
    weights -= learning_rate * gradients
    
    return (train_loss, val_loss)
```

<!-- pause -->

Rust calls this **every frame**!

<!-- end_slide -->

Python Usage
===

Simple, ergonomic API:

```python
import loss_graph
import numpy as np

# Create the live graph
graph = loss_graph.LiveGraph(
    max_epochs=100,
    title="Training Progress"
)

# Run with your training function
graph.run(train_step)
```

<!-- pause -->

Python controls training logic.
Rust handles visualization.

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

def train_step(epoch):
    global weights
    pred = weights[0] + weights[1]*X + weights[2]*X**2
    err = pred - y
    weights -= 0.1 * np.array([
        2*np.mean(err), 
        2*np.mean(err*X), 
        2*np.mean(err*X**2)
    ])
    loss = np.mean((pred - y)**2)
    return (float(loss), float(loss * 1.1))

graph = loss_graph.LiveGraph(50, "Live Demo")
graph.run(train_step)
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

1. **PyO3** enables bidirectional Rust ↔ Python
2. **Callbacks** let Python control the logic
3. **Rust** handles high-frequency rendering
4. Real-time visualization of Python training!

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
