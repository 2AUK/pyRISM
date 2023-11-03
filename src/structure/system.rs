use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(FromPyObject, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Site {
    pub atom_type: String,
    pub params: Vec<f64>,
    pub coords: Vec<f64>,
}

#[derive(FromPyObject, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Species {
    pub species_name: String,
    pub dens: f64,
    pub ns: usize,
    pub atom_sites: Vec<Site>,
}

pub struct System {
    pub species: Vec<Species>,
}

impl System {
    pub fn iter(&self) -> impl Iterator<Item = &Species> {
        self.species.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = Species> {
        self.species.into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Species> {
        self.species.iter_mut()
    }
}
