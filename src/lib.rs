use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
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

    fn start(&mut self) -> PyResult<()> {
        enable_raw_mode()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.terminal = Some(terminal);
        Ok(())
    }

    fn add_point(&mut self, epoch: f64, train_loss: f64, val_loss: f64) {
        self.train_data.push((epoch, train_loss));
        self.val_data.push((epoch, val_loss));
    }

    fn mark_complete(&mut self) {
        self.training_complete = true;
    }

    fn draw(&mut self) -> PyResult<bool> {
        let terminal = self.terminal.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Terminal not started. Call start() first.")
        })?;

        let max_loss = self
            .train_data
            .iter()
            .chain(self.val_data.iter())
            .map(|(_, y)| *y)
            .fold(0.5_f64, f64::max);

        let current_epoch = self.train_data.len();
        let max_epochs = self.max_epochs;
        let title = self.title.clone();
        let train_data = &self.train_data;
        let val_data = &self.val_data;
        let training_complete = self.training_complete;

        terminal
            .draw(|frame| {
                let area = frame.area();
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
                    "COMPLETE"
                } else {
                    "TRAINING"
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

                frame.render_widget(chart, chunks[0]);

                let legend = Block::default()
                    .title(format!(
                        " [Cyan] Training | [Yellow] Validation | Epoch {}/{} | Press 'q' to exit ",
                        current_epoch, max_epochs
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray));
                frame.render_widget(legend, chunks[1]);
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        if event::poll(Duration::from_millis(50))
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
            disable_raw_mode()
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
            let _ = disable_raw_mode();
            let _ = execute!(stdout(), LeaveAlternateScreen);
        }
    }
}

#[pymodule]
fn loss_graph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LiveGraph>()?;
    Ok(())
}
