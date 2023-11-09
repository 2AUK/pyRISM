extern crate blas_src;
pub use crate::drivers::rism::RISMDriver;
pub use crate::io::writer::RISMWriter;
pub use crate::thermodynamics::thermo::TDDriver;
use data::solution::Solutions;
use drivers::rism::{Compress, Verbosity};
use pybindings::pystructs::PyCorrelations;
use pybindings::pystructs::PyGrid;
use pybindings::pystructs::PyInteractions;
use pybindings::pystructs::PySolution;
use pybindings::pystructs::PySolvedData;
use pybindings::pystructs::PyThermodynamics;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::path::PathBuf;
use thermodynamics::thermo::Thermodynamics;

pub mod data;
pub mod drivers;
pub mod grids;
pub mod iet;
pub mod interactions;
pub mod io;
pub mod pybindings;
pub mod solvers;
pub mod structure;
pub mod thermodynamics;

#[pyclass(unsendable)]
#[pyo3(name = "Calculator")]
pub struct PyCalculator {
    pub name: String,
    verbosity: Verbosity,
    compress: Compress,
    driver: RISMDriver,
}

#[pymethods]
impl PyCalculator {
    #[new]
    pub fn new(fname: String, verbosity: String, compress: bool) -> PyResult<Self> {
        let path = PathBuf::from(&fname);
        let driver = RISMDriver::from_toml(&path);
        let name = driver.name.clone();
        let verbosity = match verbosity.as_str() {
            "quiet" => Verbosity::Quiet,
            "verbose" => Verbosity::Verbose,
            "loud" => Verbosity::VeryVerbose,
            _ => return Err(PyTypeError::new_err("Invalid verbosity flag")),
        };
        let compress = match compress {
            true => Compress::Compress,
            false => Compress::NoCompress,
        };
        Ok(PyCalculator {
            name,
            verbosity,
            compress,
            driver,
        })
    }

    pub fn execute<'py>(
        &'py mut self,
        py: Python<'py>,
    ) -> PyResult<(PySolution, PyThermodynamics, PyGrid)> {
        let solution = self.driver.execute(&self.verbosity, &self.compress);
        let grid = PyGrid::new(
            solution.config.data_config.npts,
            solution.config.data_config.radius,
            py,
        );
        let wv = self.driver.solvent.borrow().wk.clone();
        let wu = {
            match &self.driver.solute {
                Some(v) => Some(v.borrow().wk.clone()),
                None => None,
            }
        };
        let thermodynamics = TDDriver::new(&solution, wv, wu).execute();
        let _ = RISMWriter::new(&self.name, &solution, &thermodynamics).write();
        let py_corr_vv = PyCorrelations::from_correlations(solution.vv.correlations, py);
        let py_int_vv = PyInteractions::from_interactions(solution.vv.interactions, py);
        let py_vv = PySolvedData {
            interactions: py_int_vv,
            correlations: py_corr_vv,
        };

        let py_uv = match solution.uv {
            Some(uv) => {
                let py_corr_uv = PyCorrelations::from_correlations(uv.correlations, py);
                let py_int_uv = PyInteractions::from_interactions(uv.interactions, py);
                Some(PySolvedData {
                    interactions: py_int_uv,
                    correlations: py_corr_uv,
                })
            }
            None => None,
        };

        let thermodynamics = PyThermodynamics::from_thermodynamics(thermodynamics, py);

        let solution = PySolution {
            vv: py_vv,
            uv: py_uv,
        };
        Ok((solution, thermodynamics, grid))
    }
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
