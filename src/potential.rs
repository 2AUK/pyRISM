use crate::data::DataRs;
use ndarray::Array3;
use pyo3::{prelude::*, types::PyString};
use std::fmt;

#[derive(Debug, Clone)]
pub enum PotentialKind {
    LennardJones,
    HardSpheres,
    Coulomb,
    NgRenormalisation,
}

impl<'source> FromPyObject<'source> for PotentialKind {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let str = obj
            .downcast::<PyString>()?
            .to_str()
            .map(ToOwned::to_owned)
            .expect("could not convert string");
        match str.as_str() {
            "LJ" => Ok(PotentialKind::LennardJones),
            "HS" => Ok(PotentialKind::HardSpheres),
            "COU" => Ok(PotentialKind::Coulomb),
            "NG" => Ok(PotentialKind::NgRenormalisation),
            _ => panic!("not a valid potential"),
        }
    }
}

impl fmt::Display for PotentialKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PotentialKind::LennardJones => write!(f, "Lennard-Jones"),
            PotentialKind::HardSpheres => write!(f, "Hard Spheres"),
            PotentialKind::Coulomb => write!(f, "Coulomb"),
            PotentialKind::NgRenormalisation => write!(f, "Ng Renormalisation"),
        }
    }
}

#[derive(FromPyObject, Debug, Clone)]
pub struct PotentialConfig {
    pub nonbonded: PotentialKind,
    pub coulombic: PotentialKind,
    pub renormalisation: PotentialKind,
}

impl fmt::Display for PotentialConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Potential: {}", self.nonbonded)
    }
}

pub fn lennard_jones(_data: DataRs) -> Array3<f64> {
    Array3::zeros((1, 1, 1))
}
