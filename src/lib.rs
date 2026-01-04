use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
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
use std::io::{stdout, Stdout};
use std::time::Duration;

/// A live updating graph for training/validation loss powered by Ratatui in rust!
#[pyclass]
struct LiveGraph {
    train_data: Vec<(f64, f64)>,
    val_data: Vec<(f64, f64)>,
    max_epochs: usize,
    title: String,
    terminal: Option<Terminal<CrosstermBackend<Stdout>>>,
    training_complete: bool,
}

#[pymethods]
impl LiveGraph {
    #[new]
    #[pyo3(signature = (max_epochs=100, title="Training Progress"))]
    fn new(max_epochs: usize, title: &str) -> Self {
        LiveGraph {
            train_data: Vec::new(),
            val_data: Vec::new(),
            max_epochs,
            title: title.to_string(),
            terminal: None,
            training_complete: false,
        }
    }

    /// Initialize the graph to start rendering.  This takes over your terminal.
    fn start(&mut self) -> PyResult<()> {
        // Use crossterm to control the terminal directly (eg use alternate screen and render ratatui).
        crossterm::terminal::enable_raw_mode()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Setup ratatui to work with crossterm backend.
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.terminal = Some(terminal);

        Ok(())
    }

    /// Add a new point to the graph.
    fn add_point(&mut self, epoch: f64, train_loss: f64, val_loss: f64) {
        // TODO just pass 2d numpy tensor and then foreach graph and assign
        // random color or something OR have more dimensions to describe color.
        self.train_data.push((epoch, train_loss));
        self.val_data.push((epoch, val_loss));
    }

    /// Set as complete, used for display purposes.
    fn mark_complete(&mut self) {
        self.training_complete = true;
    }

    /// Draw the current graph. Returns True if the user requested to exit.
    fn draw(&mut self) -> PyResult<bool> {
        let terminal = self.terminal.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Terminal not started. Call start() first.")
        })?;

        let max_loss = self
            .train_data
            .iter()
            .chain(self.val_data.iter())
            .map(|(_, y)| *y)
            .fold(0.1_f64, f64::max);

        let current_epoch = self.train_data.len();
        let max_epochs = self.max_epochs;
        let title = self.title.clone();
        let train_data = &self.train_data;
        let val_data = &self.val_data;
        let training_complete = self.training_complete;

        // The actual Ratatui drawing code.
        terminal
            .draw(|frame| {
                let area = frame.area();

                // Splits the terminal into a vertical layout with the top graph
                // area having a min height of 10px and the bottom area having a
                // fixed hegiht of 3px (best effort).
                let chunks =
                    Layout::vertical([Constraint::Min(10), Constraint::Length(3)]).split(area);

                let x_labels = vec![
                    Span::styled("0", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{}", max_epochs / 2),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::styled(format!("{}", max_epochs), Style::default().fg(Color::Gray)),
                ];

                let y_labels = vec![
                    Span::styled("0.0", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.2}", max_loss / 2.0),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::styled(format!("{:.2}", max_loss), Style::default().fg(Color::Gray)),
                ];

                let datasets = vec![
                    Dataset::default()
                        .name("Training Loss")
                        .marker(Marker::Braille)
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(Color::Cyan))
                        .data(train_data),
                    Dataset::default()
                        .name("Validation Loss")
                        .marker(Marker::Braille)
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(Color::Yellow))
                        .data(val_data),
                ];

                let status = if training_complete {
                    "COMPLETE 🎉"
                } else {
                    "TRAINING 🤔"
                };

                let chart = Chart::new(datasets)
                    .block(
                        Block::default()
                            .title(Span::styled(
                                format!("{} [{}]", title, status),
                                Style::default().fg(Color::White).bold(),
                            ))
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::Gray)),
                    )
                    .x_axis(
                        Axis::default()
                            .title("Epoch")
                            .style(Style::default().fg(Color::Gray))
                            .bounds([0.0, max_epochs as f64])
                            .labels(x_labels),
                    )
                    .y_axis(
                        Axis::default()
                            .title("Loss")
                            .style(Style::default().fg(Color::Gray))
                            .bounds([0.0, max_loss * 1.1])
                            .labels(y_labels),
                    );

                // Render the chart to the top chunk.
                frame.render_widget(chart, chunks[0]);

                // Create the legend.
                let legend = Block::default()
                    .title(format!(
                        " [Cyan] Training | [Yellow] Validation | Epoch {}/{} | Press 'q' to exit ",
                        current_epoch, max_epochs
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray));

                // And render to the bottom chunk.
                frame.render_widget(legend, chunks[1]);
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Poll for exit.  Short if we're still training to keep UI responsive.
        if (self.training_complete
            && event::poll(Duration::from_millis(10))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?)
            || !self.training_complete
                && event::poll(Duration::from_millis(50))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        {
            if let Event::Key(key) = event::read()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            {
                if key.kind == KeyEventKind::Press
                    && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
                {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn stop(&mut self) -> PyResult<()> {
        if let Some(mut terminal) = self.terminal.take() {
            crossterm::terminal::disable_raw_mode()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            execute!(terminal.backend_mut(), LeaveAlternateScreen)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }
}

impl Drop for LiveGraph {
    fn drop(&mut self) {
        if self.terminal.is_some() {
            let _ = crossterm::terminal::disable_raw_mode();
            let _ = execute!(stdout(), LeaveAlternateScreen);
        }
    }
}

#[pymodule]
fn loss_graph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LiveGraph>()?;
    Ok(())
}
