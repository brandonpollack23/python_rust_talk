---
title: "Rust 🦀 + Python 🐍"
sub_title: Building High-Performance TUI Apps
author: Demo Presenter
---

Why Rust + Python?
===

- **Python**: Easy to use, great ecosystem
- **Rust**: Blazing fast, memory safe
- **Together**: Best of both worlds!

<!-- pause -->

Use cases:
- Performance-critical libraries
- TUI applications
- Data processing pipelines

<!-- end_slide -->

The Stack
===

```
┌─────────────────────────────┐
│         Python App          │
├─────────────────────────────┤
│     PyO3 Bindings Layer     │
├─────────────────────────────┤
│   Rust (ratatui + logic)    │
└─────────────────────────────┘
```

<!-- pause -->

**Key Technologies:**
- `PyO3` - Rust bindings for Python
- `maturin` - Build tool for PyO3
- `ratatui` - Terminal UI framework

<!-- end_slide -->

The Rust Code
===

```rust
#[pyfunction]
fn show_loss_graph(
    epochs: usize,
    initial_loss: f64,
    decay_rate: f64,
    noise_scale: f64,
    title: &str,
) -> PyResult<()> {
    run_graph(epochs, initial_loss, 
              decay_rate, noise_scale, title)
        .map_err(|e| PyRuntimeError::new_err(
            e.to_string()))
}

#[pymodule]
fn loss_graph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(show_loss_graph, m)?)?;
    Ok(())
}
```

<!-- end_slide -->

Python Usage
===

Simple, ergonomic API:

```python
import loss_graph

# One line to show the graph!
loss_graph.show_loss_graph(
    epochs=100,
    initial_loss=2.5,
    decay_rate=0.04,
    title="Training Progress"
)
```

<!-- pause -->

All parameters have sensible defaults:

```python
# This works too!
loss_graph.show_loss_graph()
```

<!-- end_slide -->

Live Demo
===

Press `Ctrl+E` to run:

```python +exec
import loss_graph
loss_graph.show_loss_graph(
    epochs=80,
    initial_loss=3.0,
    decay_rate=0.05,
    noise_scale=0.25,
    title="Live Demo - Training Loss"
)
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

1. **PyO3** makes Rust-Python integration easy
2. **ratatui** creates beautiful terminal UIs
3. **maturin** handles the build complexity
4. Python gets **native performance**

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
