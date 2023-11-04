use crate::structure::system::{Site, Species};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(FromPyObject, Debug, Clone, Serialize, Deserialize)]
pub struct ProblemConfig {
    pub temp: f64,
    pub kt: f64,
    pub ku: f64,
    pub amph: f64,
    pub drism_damping: Option<f64>,
    pub dielec: Option<f64>,
    pub nsv: usize,
    pub nsu: Option<usize>,
    pub nspv: usize,
    pub nspu: Option<usize>,
    pub npts: usize,
    pub radius: f64,
    pub nlambda: usize,
    pub preconverged: Option<PathBuf>,
    pub solvent_atoms: Vec<Site>,
    pub solute_atoms: Option<Vec<Site>>,
    pub solvent_species: Vec<Species>,
    pub solute_species: Option<Vec<Species>>,
}
