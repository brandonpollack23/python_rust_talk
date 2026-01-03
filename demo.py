#!/usr/bin/env python3
"""Demo script showing the Rust-powered loss graph visualization."""

import loss_graph

print("🚀 Launching Rust-powered TUI loss graph...")
print("   Built with ratatui + PyO3")
print("   Press 'q' to exit the visualization\n")

# Show a typical ML training loss curve
loss_graph.show_loss_graph(
    epochs=100,
    initial_loss=2.5,
    decay_rate=0.04,
    noise_scale=0.2,
    title="Neural Network Training Progress"
)

print("\n✅ Demo complete!")
