# Full Stack Explanation: loss_graph

This repository demonstrates **Rust-powered Python extensions** with **live bidirectional communication**. It builds a terminal UI (TUI) that displays a real-time ML training loss graph, where Python performs the training logic and Rust handles the visualization.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Python Script                              │
│                           (demo.py)                                 │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Training Loop (Python-controlled)                           │   │
│   │  - graph.start() → enter TUI mode                           │   │
│   │  - for epoch in range(max_epochs):                          │   │
│   │      train_step() → compute (train_loss, val_loss)          │   │
│   │      graph.add_point(epoch, train_loss, val_loss)           │   │
│   │      if graph.draw(): break  # 'q' pressed                  │   │
│   │  - graph.stop() → exit TUI mode                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                     Imperative API calls                            │
│                     (no callbacks)                                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Compiled Python Extension                        │
│                      (loss_graph.so / .pyd)                         │
│                                                                     │
│   Built by maturin from src/lib.rs                                  │
│   Uses PyO3 to expose Rust classes to Python                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Rust Core (lib.rs)                          │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ PyO3 Layer                                                  │   │
│   │  - #[pyclass] LiveGraph                                     │   │
│   │  - #[pymethods] new(), start(), add_point(), draw(),        │   │
│   │                 mark_complete(), stop()                     │   │
│   │  - Simple imperative API — Python drives the loop           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Live Graph State                                            │   │
│   │  - Vec<(f64, f64)> for train/val data                       │   │
│   │  - Terminal state (raw mode, alternate screen)              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ TUI Rendering (ratatui + crossterm)                         │   │
│   │  - start(): enters raw mode, alternate screen               │   │
│   │  - draw(): renders chart, polls for 'q' keypress            │   │
│   │  - stop(): restores terminal state                          │   │
│   │  - No event loop — Python calls draw() when ready           │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Terminal                                  │
│                                                                     │
│   crossterm: controls raw mode, alternate screen, key events        │
│   ratatui: renders widgets (Chart, Block, Axis, etc.)               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Breakdown

### `src/lib.rs` — The Rust Core

This implements the live visualization system with a simple imperative API:

1. **LiveGraph Class**
   - `#[pyclass] LiveGraph`: A Python-visible class that holds:
     - `train_data` / `val_data`: Vectors of `(epoch, loss)` points
     - `max_epochs`: Total epochs for scaling the X-axis
     - `title`: Graph title
     - `terminal`: Optional terminal state (active when in TUI mode)
   - `new(max_epochs, title)`: Constructor
   - `start()`: Enter TUI mode (raw mode, alternate screen)
   - `add_point(epoch, train_loss, val_loss)`: Add a data point
   - `draw()`: Render the chart and check for quit (returns `True` if 'q' pressed)
   - `mark_complete()`: Mark training as complete (updates title)
   - `stop()`: Exit TUI mode and restore terminal

2. **TUI Rendering**
   - `start()` enters raw mode and switches to alternate screen
   - `draw()` renders the current data and polls for 'q' keypress
   - `stop()` restores terminal state (must be called for cleanup)
   - Shows "TRAINING" or "COMPLETE" status in the title

3. **Python Bindings (PyO3)**
   - `#[pyclass]` + `#[pymethods]` expose the class to Python
   - No callbacks — Python controls the entire training loop
   - Simple return values (bool for quit detection)

### `demo.py` — Python Training Logic

Simulates a real ML training scenario:

1. **Data Generation**
   - Creates synthetic quadratic data: `y = x² + noise`
   - Separate train and validation sets

2. **Model**
   - Simple polynomial: `y = w₂x² + w₁x + w₀`
   - Starts with incorrect weights

3. **Training Loop**
   - Python controls the entire loop with `for epoch in range(...)`
   - Each iteration: compute gradients, update weights, add point to graph
   - Uses `try/finally` to ensure terminal cleanup via `graph.stop()`

4. **Visualization**
   - Creates `LiveGraph`, calls `start()` to enter TUI mode
   - Calls `add_point()` and `draw()` each epoch
   - Calls `stop()` in finally block for cleanup

### `Cargo.toml` — Rust Dependencies

| Dependency | Purpose |
|------------|---------|
| `pyo3` | Creates Python bindings. The `extension-module` feature builds a `.so`/`.pyd` that Python can import directly. |
| `ratatui` | High-level TUI framework. Provides `Chart`, `Block`, `Axis`, `Dataset` widgets. |
| `crossterm` | Low-level terminal manipulation. Handles raw mode, screen switching, and keyboard events. Cross-platform (works on Windows/macOS/Linux). |

### `pyproject.toml` — Python Build Configuration

| Field | Meaning |
|-------|---------|
| `build-backend = "maturin"` | Use maturin to compile the Rust code into a Python wheel |
| `name = "loss_graph"` | The Python package name (matches the Rust crate name) |
| `dependencies = ["numpy>=1.20"]` | Runtime dependency for the training simulation |
| `requires-python = ">=3.8"` | Minimum Python version |

---

## Build & Run Sequence

```
1. uv run maturin develop --uv
   │
   ├── maturin invokes cargo build --release
   │   └── Compiles src/lib.rs → target/release/libloss_graph.so
   │
   └── maturin copies .so into .venv/lib/python3.x/site-packages/loss_graph.cpython-*.so
       └── Python can now `import loss_graph`

2. uv run python demo.py
   │
   ├── Python imports loss_graph (loads the .so)
   ├── Python imports numpy, sets up training data
   │
   └── Python training loop:
       ├── graph.start() → enters TUI mode
       │
       └── for epoch in range(max_epochs):
           ├── train_step() → Python computes gradient, updates weights
           ├── graph.add_point(epoch, train_loss, val_loss)
           ├── graph.draw() → Rust renders, checks for 'q'
           └── break if draw() returns True
       │
       └── graph.stop() → restores terminal (in finally block)
```

---

## Data Flow During Training

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Python Training Loop                            │
│                                                                     │
│   graph.start()                                                     │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ for epoch in range(max_epochs):                             │   │
│   │     ┌──────────────┐                                        │   │
│   │     │ train_step() │ ← numpy gradient descent               │   │
│   │     └──────┬───────┘                                        │   │
│   │            │ (train_loss, val_loss)                         │   │
│   │            ▼                                                │   │
│   │     graph.add_point(epoch, train_loss, val_loss)            │   │
│   │            │                                                │   │
│   │            ▼                                                │   │
│   │     graph.draw() ──────────────────────────────────────┐    │   │
│   │            │                                           │    │   │
│   │            │ True if 'q' pressed                       ▼    │   │
│   │            │                                  ┌─────────────┐   │
│   │     if quit: break                            │    Rust     │   │
│   │                                               │  Rendering  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│        │                                          └─────────────┘   │
│        ▼                                                            │
│   graph.stop()  ← restores terminal                                 │
└─────────────────────────────────────────────────────────────────────┘
```

1. Python calls `graph.start()` to enter TUI mode
2. Python runs the training loop with `for epoch in range(...)`
3. Python computes gradients and updates weights (numpy)
4. Python calls `graph.add_point(epoch, train_loss, val_loss)`
5. Python calls `graph.draw()` — Rust renders and checks for 'q'
6. If `draw()` returns `True`, Python breaks the loop
7. Python calls `graph.stop()` in finally block to restore terminal

---

## Why This Architecture?

| Decision | Rationale |
|----------|-----------|
| **Python for training logic** | ML practitioners write Python. Keep the familiar numpy/torch workflow. |
| **Rust for visualization** | Terminal rendering needs smooth redraws. Rust's ratatui is fast and flicker-free. |
| **Python-driven loop** | Python controls when and how training happens. No surprises from Rust callbacks. |
| **Simple imperative API** | No callbacks, no closures. Just method calls that Python developers expect. |
| **try/finally pattern** | Ensures terminal cleanup even on exceptions. Familiar Python idiom. |

---

## Extending This Code

**To add more metrics:**
1. Add a third parameter to `add_point()` in Rust
2. Add another `Dataset` to the chart widget
3. Call with the additional value from Python

**To add pause/resume:**
1. Add a `paused` flag in Python
2. Handle spacebar in `draw()` and return a different signal
3. Skip training step when paused, but still call `draw()`

**To support PyTorch:**
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
