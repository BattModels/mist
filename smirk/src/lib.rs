mod gpe;
mod pre_tokenizers;
mod tokenizer;
mod wrapper;

use pyo3::prelude::*;
use tokenizer::SmirkTokenizer;

/// A Python module implemented in Rust.
#[pymodule]
fn smirk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SmirkTokenizer>()?;
    Ok(())
}

#[cfg(test)]
pub mod test_utils {
    use serde::{Deserialize, Serialize};

    /// checks that serialize and deserialize are inverse functions
    pub fn check_serde<T>(x: &T)
    where
        T: Serialize + for<'a> Deserialize<'a> + PartialEq + std::fmt::Debug,
    {
        dbg!(&x);
        let data = dbg!(serde_json::to_string::<T>(&x).unwrap());
        let loaded = dbg!(serde_json::from_str::<T>(&data).unwrap());
        assert_eq!(*x, loaded);

        // One more time in case of errors in `T`'s implementation of PartialEq
        let data_loaded = dbg!(serde_json::to_string::<T>(&loaded).unwrap());
        assert_eq!(data, data_loaded);
    }
}
