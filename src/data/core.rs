use crate::data::solution::*;
use crate::grids::radial_grid::Grid;
use crate::structure::system::{Site, Species};
use ndarray::{Array, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
#[derive(Clone, Debug)]
pub struct SystemState {
    // Thermodynamic parameters
    pub temp: f64,
    pub kt: f64,
    pub amph: f64,
    pub nlam: usize,
    pub beta: f64,
}

impl SystemState {
    pub fn new(temp: f64, kt: f64, amph: f64, nlam: usize) -> Self {
        SystemState {
            temp,
            kt,
            amph,
            nlam,
            beta: 1.0 / kt / temp,
        }
    }

    pub fn recompute_beta(&mut self) {
        self.beta = 1.0 / self.kt / self.temp;
    }
}

#[derive(Clone, Debug)]
pub struct SingleData {
    // System size
    pub sites: Vec<Site>,
    pub species: Vec<Species>,
    pub density: Array2<f64>,
    pub wk: Array3<f64>,
}

impl SingleData {
    pub fn new(sites: Vec<Site>, species: Vec<Species>, shape: (usize, usize, usize)) -> Self {
        let density = {
            let mut dens_vec: Vec<f64> = Vec::new();
            for i in species.clone().into_iter() {
                for _j in i.atom_sites {
                    dens_vec.push(i.dens);
                }
            }
            Array2::from_diag(&Array::from_vec(dens_vec))
        };

        SingleData {
            sites,
            species,
            density,
            wk: Array::zeros(shape),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Interactions {
    pub ur: Array3<f64>,
    pub u_sr: Array3<f64>,
    pub ur_lr: Array3<f64>,
    pub uk_lr: Array3<f64>,
}

impl Interactions {
    pub fn new(npts: usize, num_sites_a: usize, num_sites_b: usize) -> Self {
        let shape = (npts, num_sites_a, num_sites_b);
        Interactions {
            ur: Array::zeros(shape),
            u_sr: Array::zeros(shape),
            ur_lr: Array::zeros(shape),
            uk_lr: Array::zeros(shape),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Correlations {
    pub cr: Array3<f64>,
    pub tr: Array3<f64>,
    pub hr: Array3<f64>,
    pub hk: Array3<f64>,
}

#[derive(Clone, Debug)]
pub struct DielectricData {
    pub drism_damping: f64,
    pub diel: f64,
    pub chi: Array3<f64>,
}

impl DielectricData {
    pub fn new(drism_damping: f64, diel: f64, chi: Array3<f64>) -> Self {
        DielectricData {
            drism_damping,
            diel,
            chi,
        }
    }
}

impl Correlations {
    pub fn new(npts: usize, num_sites_a: usize, num_sites_b: usize) -> Self {
        let shape = (npts, num_sites_a, num_sites_b);
        Correlations {
            cr: Array::zeros(shape),
            tr: Array::zeros(shape),
            hr: Array::zeros(shape),
            hk: Array::zeros(shape),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataRs {
    pub system: SystemState,
    pub data_a: Rc<RefCell<SingleData>>,
    pub data_b: Rc<RefCell<SingleData>>,
    pub grid: Grid,
    pub interactions: Interactions,
    pub correlations: Correlations,
    pub dielectrics: Option<DielectricData>,
    pub solution: Option<SolvedData>,
}

impl DataRs {
    pub fn new(
        system: SystemState,
        data_a: Rc<RefCell<SingleData>>,
        data_b: Rc<RefCell<SingleData>>,
        grid: Grid,
        interactions: Interactions,
        correlations: Correlations,
        dielectrics: Option<DielectricData>,
    ) -> DataRs {
        DataRs {
            system,
            data_a,
            data_b,
            grid,
            interactions,
            correlations,
            dielectrics,
            solution: None,
        }
    }
}
