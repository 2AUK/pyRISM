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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct System {
    pub species: Vec<Species>,
    sites: Vec<Site>,
}

impl System {
    pub fn new(species: Vec<Species>) -> Self {
        let sites: Vec<Site> = species
            .iter()
            .cloned()
            .flat_map(|species| species.atom_sites)
            .collect();
        System { species, sites }
    }
    pub fn parse(input: &str) -> Self {
        println!("{}", input);
        System {
            species: Vec::new(),
            sites: Vec::new(),
        }
    }
    pub fn iter_species(&self) -> impl Iterator<Item = &Species> {
        self.species.iter()
    }

    pub fn into_iter_species(self) -> impl Iterator<Item = Species> {
        self.species.into_iter()
    }

    pub fn iter_species_mut(&mut self) -> impl Iterator<Item = &mut Species> {
        self.species.iter_mut()
    }

    pub fn iter_sites(&self) -> impl Iterator<Item = &Site> {
        self.sites.iter()
    }

    pub fn into_iter_sites(self) -> impl Iterator<Item = Site> {
        self.sites.into_iter()
    }

    pub fn iter_sites_mut(&mut self) -> impl Iterator<Item = &mut Site> {
        self.sites.iter_mut()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_system_parse_string() {
        System::parse(
            "O 78.15 3.16 -0.846 0.0 0.0 0.0
H1 7.185 1.16 0.4238 1.0 0.0 0.0
H2 7.185 1.16 0.4238 -0.3334, -0.922618, 0.0
-
Na+ 150.0 20.0 1.0 0.0 0.0 0.0
-
Cl- 200.0 50.0 -1.0 0.0 0.0 0.0",
        );
    }
}
