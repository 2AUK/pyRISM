use pyo3::prelude::*;
use pyo3::types::PyTuple;
use crate::xrism::xrism_vv_equation;
use numpy::{IntoPyArray, PyArray3};

pub mod transforms;
pub mod xrism;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_helpers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    #[pyfn(m)]
    fn xrism<'py>(py: Python<'py>) -> PyResult<&'py PyTuple> {
        todo!()
    }

    Ok(())
}
