extern crate blas_src;
pub use crate::drivers::rism::RISMDriver;
pub use crate::io::writer::RISMWriter;
pub use crate::thermodynamics::thermo::TDDriver;
use data::solution::Solutions;
use drivers::rism::{Compress, Verbosity};
use pyo3::prelude::*;
use std::path::PathBuf;
use thermodynamics::thermo::Thermodynamics;

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
    verbosity: Verbosity,
    compress: Compress,
    driver: RISMDriver,
}

impl Calculator {
    pub fn new(fname: PathBuf, verbosity: Verbosity, compress: Compress) -> Self {
        let driver = RISMDriver::from_toml(&fname);
        let name = driver.name.clone();
        Calculator {
            name,
            verbosity,
            compress,
            driver,
        }
    }

    pub fn execute(&mut self) -> (Solutions, Thermodynamics) {
        let solution = self.driver.execute(&self.verbosity, &self.compress);
        let wv = self.driver.solvent.borrow().wk.clone();
        let wu = {
            match &self.driver.solute {
                Some(v) => Some(v.borrow().wk.clone()),
                None => None,
            }
        };
        let thermodynamics = TDDriver::new(&solution, wv, wu).execute();
        let _ = RISMWriter::new(&self.name, &solution, &thermodynamics).write();
        (solution, thermodynamics)
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn librism(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCalculator>()?;
    Ok(())
}
