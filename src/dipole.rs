use crate::data::Species;
use crate::quaternion::Quaternion;
use ndarray::{Array, Array1};

pub fn total_charge(species: Vec<Species>) -> f64 {
    species.iter().fold(0.0, |acc, x| {
        acc + x
            .atom_sites
            .iter()
            .fold(0.0, |acc_inner, y| acc_inner + y.params.last().unwrap())
    })
}

pub fn centre_of_charge(species: Vec<Species>) -> Array1<f64> {
    let out = Array::zeros(3);
    out
}
