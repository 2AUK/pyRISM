use pyo3::{prelude::*, types::PyString};
use std::fmt;

// pub enum MDIISSettings {
//     depth(usize),
//     damping(f64),
// }

// pub enum GillanSettings {
//     nbasis(usize),
// }

#[derive(FromPyObject, Debug, Clone)]
pub struct SolverSettings {
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,
    pub gillan_settings: Option<usize>,
    pub mdiis_settings: Option<(usize, f64)>,
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