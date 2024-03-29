use crate::data::configuration::solver::{SolverError, SolverSettings, SolverSuccess};
use crate::data::core::DataRs;
use crate::iet::operator::Operator;
use crate::solvers::{adiis::ADIIS, lmv::LMV, mdiis::MDIIS, ng::Ng, picard::Picard};
use pyo3::{prelude::*, types::PyString};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};

pub trait Solver: Debug {
    fn solve(
        &mut self,
        data: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverKind {
    Picard,
    Ng,
    MDIIS,
    MDIIS2,
    Gillan,
    LMV,
}

impl fmt::Display for SolverKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SolverKind::Picard => write!(f, "Picard"),
            SolverKind::Ng => write!(f, "Ng"),
            SolverKind::MDIIS => write!(f, "MDIIS"),
            SolverKind::MDIIS2 => write!(f, "MDIIS2"),
            SolverKind::Gillan => write!(f, "Gillan"),
            SolverKind::LMV => write!(f, "LMV"),
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
            "MDIIS2" => Ok(SolverKind::MDIIS2),
            "Gillan" => Ok(SolverKind::Gillan),
            "LMV" => Ok(SolverKind::LMV),
            _ => panic!("not a valid solver"),
        }
    }
}

impl SolverKind {
    pub fn set(&self, settings: &SolverSettings) -> Box<dyn Solver> {
        match self {
            SolverKind::Picard => Box::new(Picard::new(settings)),
            SolverKind::MDIIS => Box::new(ADIIS::new(settings)),
            SolverKind::MDIIS2 => Box::new(MDIIS::new(settings)),
            SolverKind::Ng => Box::new(Ng::new(settings)),
            SolverKind::LMV => Box::new(LMV::new(settings)),
            _ => panic!("solver unimplemented"),
        }
    }
}
