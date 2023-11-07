use crate::interactions::potential::PotentialKind;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(FromPyObject, Debug, Clone, Serialize, Deserialize)]
pub struct PotentialConfig {
    pub nonbonded: PotentialKind,
    pub coulombic: PotentialKind,
    pub renormalisation_real: PotentialKind,
    pub renormalisation_fourier: PotentialKind,
}

impl fmt::Display for PotentialConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Potential: {}", self.nonbonded)
    }
}
