---
title: "Rust 🦀 + Python 🐍 Real World Demo"
sub_title: Live Training Visualization
author: Brandon Pollack <brandon@tokyorust.org>
---

Why Rust + Python?
===

- **Python**: Easy to use, great ML/datascience ecosystem, great as a plugin language (eg blender, Davinci, etc)
- **Rust**: Blazing fast, memory safe
- **Together**: Best of both worlds!

<!-- pause -->

Rust Use cases:
<!-- pause -->
- Awesome systems libraries
<!-- pause -->
- Performance-critical libraries
<!-- pause -->
- Real-time visualizations

<!-- end_slide -->

The Stack
===

```
┌───────────────────────────── ┐
│   Python (numpy for training)|
├───────────────────────────── ┤
│     PyO3 Bindings Layer      │
├───────────────────────────── ┤
│   Rust (ratatui live TUI)    │
└───────────────────────────── ┘
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
Why would you do this?
===

You wouldn't necessarily do it exactly like this, I'm just trying to keep it simple.

<!-- pause -->

In reality there are more robust graphing libraries like the one that comes with [burn](burn.dev),
an ML framework with swappable backends (libtorch, wgpu, etc) built directly in rust.

Other things you may want to consider using from the rust ecosystem but driving from python are:

<!-- pause -->

* [bevy](bevy.rs) -- A data oriented game engine in rust
* [tokio](tokio.rs) -- An amazing async runtime in rust
* Rayon -- A super fast and safe multithreading/parallelism runtime
* Serialization


<!-- end_slide -->

Python Training Logic
===

For this example we're just training regression
```latex +render
$$f(x) = w_2x^2 + w_1x + w_0$$
```


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
```python
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
```

<!-- end_slide -->

Live Demo
===

Press `Ctrl+E` to run:

```bash +exec +acquire_terminal
uv run python demo.py
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
