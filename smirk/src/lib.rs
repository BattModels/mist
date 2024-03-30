mod pretokenizer;
mod split_selfies;
mod split_smiles;
mod tokenizer;

use pyo3::{prelude::*, types::PyString};
use split_selfies::split_chemically_consistent as selfies_split_chemically_consistent;
use split_smiles::split_chemically_consistent as smiles_split_chemically_consistent;
use tokenizer::SmirkTokenizer;

#[pyfunction]
fn chemically_consistent_split(a: &PyString, is_smiles: bool) -> PyResult<Vec<String>> {
    if is_smiles {
        Ok(smiles_split_chemically_consistent(a.to_str().unwrap()))
    } else {
        Ok(selfies_split_chemically_consistent(a.to_str().unwrap()))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn smirk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chemically_consistent_split, m)?)?;
    m.add_class::<SmirkTokenizer>()?;
    Ok(())
}
