# Full Stack Explanation: loss_graph

This repository demonstrates **Rust-powered Python extensions**. It builds a terminal UI (TUI) that displays a simulated ML training loss graph, written in Rust but callable from Python.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Python Script                              │
│                           (demo.py)                                 │
│                               │                                     │
│                     import loss_graph                               │
│                     loss_graph.show_loss_graph(...)                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Compiled Python Extension                        │
│                      (loss_graph.so / .pyd)                         │
│                                                                     │
│   Built by maturin from src/lib.rs                                  │
│   Uses PyO3 to expose Rust functions to Python                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Rust Core (lib.rs)                          │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ PyO3 Layer                                                  │   │
│   │  - #[pyfunction] show_loss_graph(...)                       │   │
│   │  - #[pymodule] loss_graph                                   │   │
│   │  - Converts Rust errors → Python exceptions                 │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Application Logic                                           │   │
│   │  - generate_loss_data(): creates fake ML loss curve         │   │
│   │  - exponential_decay_loss(): simulates loss with noise      │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ TUI Rendering (ratatui + crossterm)                         │   │
│   │  - run_graph(): main event loop                             │   │
│   │  - Renders a Chart widget with two datasets                 │   │
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

This is the entire application logic. It does three things:

1. **Data Generation**
   - `exponential_decay_loss()`: Computes a single loss value using exponential decay + sinusoidal noise
   - `generate_loss_data()`: Generates a vector of `(epoch, loss)` tuples for plotting

2. **TUI Rendering**
   - `run_graph()`: The main function that:
     - Enables terminal raw mode (captures all keypresses)
     - Switches to an alternate screen buffer (so the graph doesn't pollute your shell history)
     - Creates a `ratatui::Terminal`
     - Enters an event loop: draws the chart, waits for keypress, exits on `q` or `Esc`
     - Restores terminal state on exit

3. **Python Bindings (PyO3)**
   - `#[pyfunction] show_loss_graph(...)`: The function Python sees. Wraps `run_graph()` and converts Rust `io::Error` into Python `RuntimeError`
   - `#[pymodule] loss_graph`: Declares the Python module and registers `show_loss_graph`

### `Cargo.toml` — Rust Dependencies

| Dependency | Purpose |
|------------|---------|
| `pyo3` | Creates Python bindings. The `extension-module` feature builds a `.so`/`.pyd` that Python can import directly. |
| `ratatui` | High-level TUI framework. Provides `Chart`, `Block`, `Axis`, `Dataset` widgets. |
| `crossterm` | Low-level terminal manipulation. Handles raw mode, screen switching, and keyboard events. Cross-platform (works on Windows/macOS/Linux). |

The `crate-type = ["cdylib", "rlib"]` line tells Rust to produce:
- `cdylib`: A C-compatible dynamic library (this becomes the Python module)
- `rlib`: A Rust library (for testing/linking within Rust)

### `pyproject.toml` — Python Build Configuration

This file tells Python how to build the project:

| Field | Meaning |
|-------|---------|
| `build-backend = "maturin"` | Use maturin to compile the Rust code into a Python wheel |
| `name = "loss_graph"` | The Python package name (matches the Rust crate name) |
| `requires-python = ">=3.8"` | Minimum Python version |

When you run `maturin develop`, it:
1. Runs `cargo build` to compile the Rust code
2. Packages the resulting `.so` file as a Python extension
3. Installs it into your virtual environment

### `demo.py` — Python Entry Point

A simple script that imports the Rust module and calls `show_loss_graph()`. This demonstrates that from Python's perspective, the Rust code looks like any other Python module.

### `mise.toml` — Development Tool Versions

`mise` is a polyglot version manager. This file declares:
- `uv`: Python package manager (faster alternative to pip)
- `pipx`: For installing Python CLI tools in isolation
- `specify-cli`: Unknown/project-specific tool
- `amp`: Unknown/project-specific tool

You install mise, then `mise install` gives you the correct tool versions.

### `uv.lock` — Python Dependency Lock

Generated by `uv`. Pins exact versions of Python dependencies for reproducibility.

### `Cargo.lock` — Rust Dependency Lock

Generated by Cargo. Pins exact versions of Rust dependencies for reproducibility.

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
   │
   └── Calls loss_graph.show_loss_graph(...)
       │
       └── Rust takes over the terminal, draws the chart, waits for 'q'
```

---

## What Each Technology Does

| Technology | Role |
|------------|------|
| **Rust** | The implementation language. Provides memory safety, speed, and a rich ecosystem. |
| **PyO3** | A Rust crate that provides bindings between Rust and Python. It handles type conversion, the GIL, and exception propagation. |
| **maturin** | A build tool that compiles Rust code into Python wheels. Handles the complex linking and packaging. |
| **ratatui** | A Rust TUI framework (fork of `tui-rs`). Provides declarative widgets for terminal UIs. |
| **crossterm** | A Rust crate for cross-platform terminal manipulation. Handles raw mode, colors, and input events. |
| **uv** | A fast Python package manager. Replaces pip/venv. Creates `.venv/` and manages dependencies. |
| **mise** | A tool version manager (like asdf/nvm). Ensures everyone uses the same tool versions. |

---

## Data Flow When Running

1. Python calls `loss_graph.show_loss_graph(epochs=100, ...)`
2. PyO3 converts Python arguments to Rust types
3. `run_graph()` is called:
   - Generates training data: 100 `(epoch, loss)` points with decay + noise
   - Generates validation data: similar but with 10% higher initial loss, slower decay, more noise
   - Enables raw mode (terminal no longer line-buffers input)
   - Switches to alternate screen (new blank terminal buffer)
   - Enters render loop:
     - Clears screen
     - Draws `Chart` widget with two `Dataset`s (training = cyan, validation = yellow)
     - Draws legend bar at bottom
     - Waits for keypress
     - If `q` or `Esc`, break loop
   - Disables raw mode
   - Returns to normal screen
4. PyO3 converts Rust `Result` to Python (raises exception on error)
5. Python continues (`print("✅ Demo complete!")`)

---

## Why This Architecture?

| Decision | Rationale |
|----------|-----------|
| **Rust for the TUI** | Terminal rendering is performance-sensitive (smooth redraws). Rust's `ratatui` is mature and fast. |
| **Python bindings** | ML practitioners use Python. This lets them call the visualizer from their training scripts without leaving Python. |
| **PyO3 + maturin** | The standard way to build Rust → Python extensions. Handles all the ABI complexity. |
| **Single function API** | Simple is better. One function with sensible defaults makes adoption trivial. |

---

## Extending This Code

To add a new Python-callable function:

1. Add `#[pyfunction]` to a Rust function in `lib.rs`
2. Register it in the `#[pymodule]` block: `m.add_function(wrap_pyfunction!(your_fn, m)?)?;`
3. Rebuild: `uv run maturin develop --uv`

To add new chart types or widgets, use `ratatui`'s widget system in `run_graph()`.
