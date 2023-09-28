use crate::data::DataRs;
use crate::mdiis::MDIIS;
use crate::operator::Operator;
use pyo3::{prelude::*, types::PyString};
use std::fmt::{self, Debug, Display};

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
    fn solve(&mut self, data: &mut DataRs, operator: &Operator) -> Result<(), SolverError>;
}

#[derive(FromPyObject, Debug, Clone)]
pub struct MDIISSettings {
    pub depth: usize,
    pub damping: f64,
}

#[derive(FromPyObject, Debug, Clone)]
pub struct GillanSettings {
    pub nbasis: usize,
}

#[derive(FromPyObject, Debug, Clone)]
pub struct SolverSettings {
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,
    pub gillan_settings: Option<GillanSettings>,
    pub mdiis_settings: Option<MDIISSettings>,
}

#[derive(FromPyObject, Debug, Clone)]
pub struct SolverConfig {
    pub solver: SolverKind,
    pub settings: SolverSettings,
}

impl fmt::Display for SolverConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Solver: {}", self.solver)
    }
}

#[derive(Debug, Clone)]
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
    pub fn set(&self, settings: &SolverSettings) -> impl Solver {
        match self {
            SolverKind::MDIIS => MDIIS::new(settings),
            _ => panic!("solver unimplemented"),
        }
    }
}
