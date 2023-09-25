extern crate blas_src;
use crate::driver::RISMDriver;
use pyo3::prelude::*;

pub mod closure;
pub mod data;
pub mod driver;
pub mod integralequation;
pub mod mdiis;
pub mod operator;
pub mod potential;
pub mod solver;
pub mod transforms;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_helpers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RISMDriver>()?;
    Ok(())
}
