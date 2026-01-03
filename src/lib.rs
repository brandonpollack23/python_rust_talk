use pyo3::prelude::*;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Color, Style, Stylize},
    symbols::Marker,
    text::Span,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType},
    Terminal,
};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::{self, stdout};

fn exponential_decay_loss(epoch: f64, initial: f64, decay_rate: f64, noise_scale: f64) -> f64 {
    let base = initial * (-decay_rate * epoch).exp();
    let noise = noise_scale * (epoch * 7.3).sin() * (-epoch * 0.1).exp();
    (base + noise).max(0.01)
}

fn generate_loss_data(epochs: usize, initial: f64, decay_rate: f64, noise_scale: f64) -> Vec<(f64, f64)> {
    (0..epochs)
        .map(|e| {
            let epoch = e as f64;
            (epoch, exponential_decay_loss(epoch, initial, decay_rate, noise_scale))
        })
        .collect()
}

fn run_graph(
    epochs: usize,
    initial_loss: f64,
    decay_rate: f64,
    noise_scale: f64,
    title: &str,
) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let train_data = generate_loss_data(epochs, initial_loss, decay_rate, noise_scale);
    let val_data = generate_loss_data(epochs, initial_loss * 1.1, decay_rate * 0.9, noise_scale * 1.5);

    let max_loss = train_data.iter()
        .chain(val_data.iter())
        .map(|(_, y)| *y)
        .fold(0.0_f64, f64::max);

    loop {
        terminal.draw(|frame| {
            let area = frame.area();
            
            let chunks = Layout::vertical([
                Constraint::Min(10),
                Constraint::Length(3),
            ]).split(area);

            let x_labels = vec![
                Span::styled("0", Style::default().fg(Color::Gray)),
                Span::styled(format!("{}", epochs / 2), Style::default().fg(Color::Gray)),
                Span::styled(format!("{}", epochs), Style::default().fg(Color::Gray)),
            ];

            let y_labels = vec![
                Span::styled("0.0", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.1}", max_loss / 2.0), Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.1}", max_loss), Style::default().fg(Color::Gray)),
            ];

            let datasets = vec![
                Dataset::default()
                    .name("Training Loss")
                    .marker(Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&train_data),
                Dataset::default()
                    .name("Validation Loss")
                    .marker(Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Yellow))
                    .data(&val_data),
            ];

            let chart = Chart::new(datasets)
                .block(
                    Block::default()
                        .title(Span::styled(title, Style::default().fg(Color::White).bold()))
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::Gray)),
                )
                .x_axis(
                    Axis::default()
                        .title("Epoch")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([0.0, epochs as f64])
                        .labels(x_labels),
                )
                .y_axis(
                    Axis::default()
                        .title("Loss")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([0.0, max_loss * 1.1])
                        .labels(y_labels),
                );

            frame.render_widget(chart, chunks[0]);

            let legend = Block::default()
                .title(" [Cyan] Training | [Yellow] Validation | Press 'q' to exit ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray));
            frame.render_widget(legend, chunks[1]);
        })?;

        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc) {
                break;
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (epochs=100, initial_loss=2.5, decay_rate=0.05, noise_scale=0.15, title="Training Loss Over Time"))]
fn show_loss_graph(
    epochs: usize,
    initial_loss: f64,
    decay_rate: f64,
    noise_scale: f64,
    title: &str,
) -> PyResult<()> {
    run_graph(epochs, initial_loss, decay_rate, noise_scale, title)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn loss_graph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(show_loss_graph, m)?)?;
    Ok(())
}
