extern crate blas_src;
use crate::driver::RISMDriver;
use pyo3::prelude::*;

pub mod closure;
pub mod data;
pub mod dipole;
pub mod driver;
pub mod input;
pub mod integralequation;
pub mod mdiis;
pub mod ng;
pub mod operator;
pub mod picard;
pub mod potential;
pub mod quaternion;
pub mod solution;
pub mod solver;
pub mod thermo;
pub mod transforms;
pub mod writer;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn librism(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RISMDriver>()?;
    Ok(())
}
