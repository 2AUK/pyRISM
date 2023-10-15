use crate::data::DataRs;
use crate::mdiis::MDIIS;
use crate::ng::Ng;
use crate::operator::Operator;
use crate::picard::Picard;
use pyo3::{prelude::*, types::PyString};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug, Display};

#[derive(Debug)]
pub struct SolverSuccess(pub usize, pub f64);

impl Display for SolverSuccess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Finished! Iteration: {} RMSE: {:.6E}", self.0, self.1)
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

pub trait Solver: Debug {
    fn solve(
        &mut self,
        data: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError>;
}

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

impl fmt::Display for SolverConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Solver: {}", self.solver)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverKind {
    Picard,
    Ng,
    MDIIS,
    Gillan,
}

impl fmt::Display for SolverKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SolverKind::Picard => write!(f, "Picard"),
            SolverKind::Ng => write!(f, "Ng"),
            SolverKind::MDIIS => write!(f, "MDIIS"),
            SolverKind::Gillan => write!(f, "Gillan"),
        }
    }
}

impl<'source> FromPyObject<'source> for SolverKind {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let str = obj
            .downcast::<PyString>()?
            .to_str()
            .map(ToOwned::to_owned)
            .expect("could not convert string");
        match str.as_str() {
            "Picard" => Ok(SolverKind::Picard),
            "Ng" => Ok(SolverKind::Ng),
            "MDIIS" => Ok(SolverKind::MDIIS),
            "Gillan" => Ok(SolverKind::Gillan),
            _ => panic!("not a valid solver"),
        }
    }
}

impl SolverKind {
    pub fn set(&self, settings: &SolverSettings) -> Box<dyn Solver> {
        match self {
            SolverKind::Picard => Box::new(Picard::new(settings)),
            SolverKind::MDIIS => Box::new(MDIIS::new(settings)),
            SolverKind::Ng => Box::new(Ng::new(settings)),
            _ => panic!("solver unimplemented"),
        }
    }
}
