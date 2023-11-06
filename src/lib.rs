extern crate blas_src;
pub use crate::drivers::rism::RISMDriver;
use pyo3::prelude::*;
use std::path::PathBuf;

pub mod data;
pub mod drivers;
pub mod grids;
pub mod iet;
pub mod interactions;
pub mod io;
pub mod solvers;
pub mod structure;
pub mod thermodynamics;

#[pyclass(unsendable)]
#[pyo3(name = "Calculator")]
pub struct PyCalculator {
    pub name: String,
    driver: RISMDriver,
}

pub struct Calculator {
    pub name: String,
    driver: RISMDriver,
}

impl Calculator {
    pub fn from_toml(fname: PathBuf) -> Self {
        let driver = RISMDriver::from_toml(&fname);
        let name = driver.name.clone();
        Calculator { name, driver }
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn librism(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RISMDriver>()?;
    Ok(())
}
