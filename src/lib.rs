//! # pyRISM
//!
//! pyRISM is a tool for solving the Reference Interaction Site Model equation and its variants.
//! The tool was originally developed in Python (hence pyRISM) but has since been rewritten in
//! Rust. This has resulted in speed-ups and better developer ergonomics (in my opinion, anyway).
//!
//! The purpose of this documentation is to make clear the design decisions as a reference for
//! myself and for other scientific software developers interested in RISM and the numerics.
//! Currently, this is a work in progress.
//!
//! This documentation encompasses the library portion of the code, named `librism` and is where
//! a majority of the implementation resides.
//!
//! Briefly, pyRISM solves the Fourier-space matrix equation:
//!
//! $$ H = \omega C \omega + \rho \omega C H $$
//!
//! for unknowns $C$ and $H$. Since there are 2 unknowns, the equations need to be "closed" with an
//! aptly named closure:
//!
//! $$ c(r) = \exp(-\beta U(r) + t(r) + B(r)) - 1 - t(r) $$
//!
//! where $t(r) = h(r) - c(r)$.

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

/// The Python binding for the Calculator struct, used when pyRISM is called from a Python script
/// as a module.
#[pyclass(unsendable)]
#[pyo3(name = "Calculator")]
pub struct PyCalculator {
    /// Path to input file
    pub name: String,
    /// Verbosity switch
    pub verbosity: Verbosity,
    /// Solvent-solvent problem compression switch
    pub compress: Compress,
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
        let (solution, diagnostics) = self.driver.execute(&self.verbosity, &self.compress);
        let grid = PyGrid::new(
            solution.config.data_config.npts,
            solution.config.data_config.radius,
            py,
        );
        let wv = self.driver.solvent.borrow().wk.clone();
        let wu = self.driver.solute.as_ref().map(|v| v.borrow().wk.clone());
        let thermodynamics = TDDriver::new(&solution, wv, wu).execute();
        let _ = RISMWriter::new(&self.name, &solution, &thermodynamics, &diagnostics).write();
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

/// The Calculator struct controls the external parameters to the Driver structs. As of now, it
/// simply processes verbosity and compression switches that are further passed into the various
/// Drivers.
pub struct Calculator {
    /// Path to input file
    pub name: String,
    /// Verbosity switch
    pub verbosity: Verbosity,
    /// Solvent-solvent problem compression switch
    pub compress: Compress,
    driver: RISMDriver,
}

impl Calculator {
    /// Construct a new driver with the correct compression and verbosity flags
    pub fn new(fname: PathBuf, verbosity: Verbosity, compress: Compress) -> Self {
        // Pull problem information from .toml file
        let driver = RISMDriver::from_toml(&fname);
        let name = driver.name.clone();
        Calculator {
            name,
            verbosity,
            compress,
            driver,
        }
    }

    /// Start the Driver to solve the problem, run the TDDriver to compute
    /// thermodynamic properties from the solved problem, and finally write all the outputs via
    /// RISMWriter.
    pub fn execute(&mut self) -> (Solutions, Thermodynamics) {
        let (solution, diagnostics) = self.driver.execute(&self.verbosity, &self.compress);

        // Pull the solvent and solute intramolecular correlation functions to be used by TDDriver
        let wv = self.driver.solvent.borrow().wk.clone();
        let wu = self.driver.solute.as_ref().map(|v| v.borrow().wk.clone());

        let thermodynamics = TDDriver::new(&solution, wv, wu).execute();

        let _ = RISMWriter::new(&self.name, &solution, &thermodynamics, &diagnostics).write();

        // Return the solutions and thermodynamics if being used as a crate
        (solution, thermodynamics)
    }
}

/// The `librism` Python module.
#[pymodule]
fn librism(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCalculator>()?;
    Ok(())
}
