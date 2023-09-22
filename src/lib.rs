extern crate blas_src;
use crate::driver::RISMDriver;
use pyo3::prelude::*;

pub mod closure;
pub mod driver;
pub mod data;
pub mod mdiis;
pub mod transforms;
pub mod xrism;
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_helpers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RISMDriver>()?;
    Ok(())
}
