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
use std::io::stdout;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[pyclass]
struct LiveGraph {
    train_data: Arc<Mutex<Vec<(f64, f64)>>>,
    val_data: Arc<Mutex<Vec<(f64, f64)>>>,
    max_epochs: usize,
    title: String,
}

#[pymethods]
impl LiveGraph {
    #[new]
    #[pyo3(signature = (max_epochs=100, title="Training Progress"))]
    fn new(max_epochs: usize, title: &str) -> Self {
        LiveGraph {
            train_data: Arc::new(Mutex::new(Vec::new())),
            val_data: Arc::new(Mutex::new(Vec::new())),
            max_epochs,
            title: title.to_string(),
        }
    }

    fn add_point(&self, epoch: f64, train_loss: f64, val_loss: f64) {
        self.train_data.lock().unwrap().push((epoch, train_loss));
        self.val_data.lock().unwrap().push((epoch, val_loss));
    }

    fn run(&self, py: Python<'_>, train_step_fn: PyObject) -> PyResult<()> {
        enable_raw_mode().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut current_epoch = 0usize;
        let mut training_complete = false;

        loop {
            if !training_complete && current_epoch < self.max_epochs {
                let result = train_step_fn.call1(py, (current_epoch,))?;
                let (train_loss, val_loss): (f64, f64) = result.extract(py)?;
                self.add_point(current_epoch as f64, train_loss, val_loss);
                current_epoch += 1;
                if current_epoch >= self.max_epochs {
                    training_complete = true;
                }
            }

            let train_data = self.train_data.lock().unwrap().clone();
            let val_data = self.val_data.lock().unwrap().clone();

            let max_loss = train_data
                .iter()
                .chain(val_data.iter())
                .map(|(_, y)| *y)
                .fold(0.5_f64, f64::max);

            let title = self.title.clone();
            let max_epochs = self.max_epochs;

            terminal
                .draw(|frame| {
                    let area = frame.area();
                    let chunks =
                        Layout::vertical([Constraint::Min(10), Constraint::Length(3)]).split(area);

                    let x_labels = vec![
                        Span::styled("0", Style::default().fg(Color::Gray)),
                        Span::styled(format!("{}", max_epochs / 2), Style::default().fg(Color::Gray)),
                        Span::styled(format!("{}", max_epochs), Style::default().fg(Color::Gray)),
                    ];

                    let y_labels = vec![
                        Span::styled("0.0", Style::default().fg(Color::Gray)),
                        Span::styled(format!("{:.2}", max_loss / 2.0), Style::default().fg(Color::Gray)),
                        Span::styled(format!("{:.2}", max_loss), Style::default().fg(Color::Gray)),
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
                if let Event::Key(key) =
                    event::read().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                {
                    if key.kind == KeyEventKind::Press
                        && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
                    {
                        break;
                    }
                }
            }
        }

        disable_raw_mode().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}

#[pymodule]
fn loss_graph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LiveGraph>()?;
    Ok(())
}
