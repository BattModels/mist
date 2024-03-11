mod split;

use pyo3::{prelude::*, types::PyString};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn chemically_consistent_split(a: &PyString) -> PyResult<String> {
    Ok(a.to_string())
}


/// A Python module implemented in Rust.
#[pymodule]
fn smirk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chemically_consistent_split, m)?)?;
    Ok(())
}
