# Rust + Python TUI Demo

A demonstration of using Rust with ratatui to create a terminal-based loss graph visualization, exposed to Python via PyO3.

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

Press `q` to exit the visualization.

### 3. Run the presentation

```bash
presenterm presentation.md
```

Use arrow keys to navigate. On slides with executable code, press `Ctrl+E` to run the code.

## Project Structure

```
.
├── Cargo.toml          # Rust dependencies
├── pyproject.toml      # Python build config
├── src/
│   └── lib.rs          # Rust library with PyO3 bindings
├── demo.py             # Python demo script
├── presentation.md     # presenterm slides
└── instructions.md     # This file
```

## How It Works

1. **Rust Core** (`src/lib.rs`): Implements the TUI graph using ratatui, simulating a typical ML training loss curve with exponential decay and noise.

2. **PyO3 Bindings**: The `show_loss_graph` function is exposed to Python with configurable parameters:
   - `epochs`: Number of training epochs (default: 100)
   - `initial_loss`: Starting loss value (default: 2.5)
   - `decay_rate`: How fast the loss decreases (default: 0.05)
   - `noise_scale`: Amount of noise in the curve (default: 0.15)
   - `title`: Graph title

3. **Python Interface**: Simple one-liner to display the graph:
   ```python
   import loss_graph
   loss_graph.show_loss_graph()
   ```

## API Reference

```python
import loss_graph

# All parameters are optional with sensible defaults
loss_graph.show_loss_graph(
    epochs=100,
    initial_loss=2.5,
    decay_rate=0.05,
    noise_scale=0.15,
    title="Training Loss Over Time"
)
```
