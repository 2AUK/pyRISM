use crate::solvers::solver::SolverKind;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[derive(Debug)]
pub struct SolverSuccess(pub usize, pub f64, pub f64);

impl Display for SolverSuccess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Finished! Iteration: {} RMSE: {:.6E} Time Elapsed: {}",
            self.0, self.1, self.2
        )
    }
}

#[derive(Debug)]
pub enum SolverError {
    ConvergenceError(usize),
    MaxIterationError(usize),
}

impl Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::ConvergenceError(i) => write!(f, "Solver diverged at iteration {}", i),
            SolverError::MaxIterationError(i) => write!(f, "Max iteration reach at {}", i),
        }
    }
}

impl std::error::Error for SolverError {}

#[derive(FromPyObject, Debug, Clone, Serialize, Deserialize)]
pub struct MDIISSettings {
    pub depth: usize,
    pub damping: f64,
}

impl Display for MDIISSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "├Depth: {}\n└MDIIS Damping: {}",
            self.depth, self.damping
        )
    }
}

#[derive(FromPyObject, Debug, Clone, Serialize, Deserialize)]
pub struct GillanSettings {
    pub nbasis: usize,
}

impl Display for GillanSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "└Num. Basis: {}", self.nbasis)
    }
}
#[derive(FromPyObject, Debug, Clone, Serialize, Deserialize)]
pub struct SolverSettings {
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,
    pub gillan_settings: Option<GillanSettings>,
    pub mdiis_settings: Option<MDIISSettings>,
}

#[derive(FromPyObject, Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub solver: SolverKind,
    pub settings: SolverSettings,
}

impl Display for SolverConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Solver: {}", self.solver)
    }
}
