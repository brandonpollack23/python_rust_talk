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
│   │ Training Logic (numpy)                                      │   │
│   │  - Polynomial regression with gradient descent              │   │
│   │  - train_step(epoch) → (train_loss, val_loss)               │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                     graph = LiveGraph(...)                          │
│                     graph.run(train_step)                           │
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
│   │  - #[pymethods] new(), add_point(), run()                   │   │
│   │  - Calls Python callback each frame for new data            │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Live Graph State                                            │   │
│   │  - Arc<Mutex<Vec<(f64, f64)>>> for train/val data           │   │
│   │  - Thread-safe accumulation of loss points                  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ TUI Rendering (ratatui + crossterm)                         │   │
│   │  - run(): main event loop with 50ms poll timeout            │   │
│   │  - Calls Python train_step_fn each iteration                │   │
│   │  - Renders Chart widget with growing datasets               │   │
│   │  - Handles keyboard input (q/Esc to quit)                   │   │
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

This implements the live visualization system with bidirectional Python communication:

1. **LiveGraph Class**
   - `#[pyclass] LiveGraph`: A Python-visible class that holds:
     - `train_data` / `val_data`: Thread-safe vectors of `(epoch, loss)` points
     - `max_epochs`: Total epochs to run
     - `title`: Graph title
   - `new()`: Constructor with optional parameters
   - `add_point()`: Manually add a data point
   - `run(train_step_fn)`: Main entry point that:
     - Takes a Python callable as argument
     - Calls it each frame with the current epoch
     - Extracts `(train_loss, val_loss)` from the return value
     - Updates the graph in real-time

2. **TUI Rendering**
   - Uses `event::poll(Duration::from_millis(50))` for non-blocking input
   - Each iteration: call Python → add point → redraw → check for quit
   - Shows "TRAINING" or "COMPLETE" status in the title

3. **Python Bindings (PyO3)**
   - `#[pyclass]` + `#[pymethods]` expose the class to Python
   - `PyObject` accepts any Python callable
   - `call1(py, (epoch,))` invokes Python with arguments
   - `.extract::<(f64, f64)>(py)` converts Python tuple to Rust

### `demo.py` — Python Training Logic

Simulates a real ML training scenario:

1. **Data Generation**
   - Creates synthetic quadratic data: `y = x² + noise`
   - Separate train and validation sets

2. **Model**
   - Simple polynomial: `y = w₂x² + w₁x + w₀`
   - Starts with incorrect weights

3. **Training Step**
   - Computes analytical gradients for MSE loss
   - Performs SGD update with decaying noise
   - Returns `(train_loss, val_loss)` tuple

4. **Visualization**
   - Creates `LiveGraph` and passes `train_step` callback
   - Rust calls Python each frame, Python updates weights and returns losses

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
   └── Calls graph.run(train_step)
       │
       └── Rust event loop:
           ├── Call train_step(epoch) → Python computes gradient, updates weights
           ├── Python returns (train_loss, val_loss)
           ├── Rust adds point to graph, redraws
           └── Repeat until max_epochs or 'q' pressed
```

---

## Data Flow During Training

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Rust      │  call   │   Python    │ compute │   NumPy     │
│  LiveGraph  │ ──────► │ train_step  │ ──────► │  Gradient   │
│   .run()    │         │   (epoch)   │         │   Descent   │
└─────────────┘         └─────────────┘         └─────────────┘
      ▲                       │
      │                       │ return (train_loss, val_loss)
      │                       ▼
      │                 ┌─────────────┐
      │                 │  Tuple      │
      │                 │ Extraction  │
      │                 └─────────────┘
      │                       │
      └───────────────────────┘
              add_point()
              redraw chart
```

1. Rust calls `train_step_fn.call1(py, (epoch,))`
2. Python executes gradient descent step using numpy
3. Python returns `(train_loss, val_loss)` tuple
4. PyO3 extracts tuple to Rust `(f64, f64)`
5. Rust calls `add_point()` to store the data
6. ratatui redraws the chart with updated data
7. Loop continues until `max_epochs` reached

---

## Why This Architecture?

| Decision | Rationale |
|----------|-----------|
| **Python for training logic** | ML practitioners write Python. Keep the familiar numpy/torch workflow. |
| **Rust for visualization** | Terminal rendering needs smooth redraws. Rust's ratatui is fast and flicker-free. |
| **Callback-based design** | Python stays in control of the training loop logic. Rust just asks for data. |
| **Real-time updates** | More engaging demo. Shows true bidirectional communication, not just "Rust generates data". |

---

## Extending This Code

**To add more metrics:**
1. Change `train_step` to return more values: `(train_loss, val_loss, accuracy)`
2. Update Rust to extract a 3-tuple and add a third dataset
3. Add another `Dataset` to the chart

**To add pause/resume:**
1. Handle spacebar in Rust's event loop
2. Skip calling `train_step_fn` when paused
3. Update status display

**To support PyTorch:**
```python
def train_step(epoch: int) -> tuple[float, float]:
    optimizer.zero_grad()
    loss = model(X_train).loss()
    loss.backward()
    optimizer.step()
    return (loss.item(), compute_val_loss())
```
