use crate::solution::*;
use ndarray::{Array, Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::f64::consts::PI;
use std::path::PathBuf;
use std::rc::Rc;

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

#[derive(FromPyObject, Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
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

#[derive(Clone, Debug)]
pub struct Grid {
    pub npts: usize,
    pub radius: f64,
    pub dr: f64,
    pub dk: f64,
    pub rgrid: Array1<f64>,
    pub kgrid: Array1<f64>,
}

impl Grid {
    pub fn new(npts: usize, radius: f64) -> Self {
        let dr = radius / npts as f64;
        let dk = 2.0 * PI / (2.0 * npts as f64 * dr);
        Grid {
            npts,
            radius,
            dr,
            dk,
            rgrid: Array1::range(0.5, npts as f64, 1.0) * dr,
            kgrid: Array1::range(0.5, npts as f64, 1.0) * dk,
        }
    }
}

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

#[pyclass]
#[derive(Clone)]
pub struct DataPy {
    #[pyo3(get, set)]
    temp: f64,
    #[pyo3(get, set)]
    kt: f64,
    #[pyo3(get, set)]
    ku: f64,
    #[pyo3(get, set)]
    amph: f64,
    #[pyo3(get, set)]
    ns1: usize,
    #[pyo3(get, set)]
    ns2: usize,
    #[pyo3(get, set)]
    nsp1: usize,
    #[pyo3(get, set)]
    nsp2: usize,
    #[pyo3(get, set)]
    npts: usize,
    #[pyo3(get, set)]
    radius: f64,
    #[pyo3(get, set)]
    nlam: usize,

    #[pyo3(get, set)]
    pub cr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub tr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub hr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub hk: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub gr: Py<PyArray3<f64>>,

    #[pyo3(get, set)]
    beta: f64,
    #[pyo3(get, set)]
    pub ur: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub u_sr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub ur_lr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub uk_lr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub wk: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub density: Py<PyArray2<f64>>,

    #[pyo3(get, set)]
    pub rgrid: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    pub kgrid: Py<PyArray1<f64>>,
}

#[pymethods]
impl DataPy {
    #[new]
    fn new(
        temp: f64,
        kt: f64,
        ku: f64,
        amph: f64,
        ns1: usize,
        ns2: usize,
        nsp1: usize,
        nsp2: usize,
        npts: usize,
        radius: f64,
        nlam: usize,
        cr: Py<PyArray3<f64>>,
        tr: Py<PyArray3<f64>>,
        hr: Py<PyArray3<f64>>,
        hk: Py<PyArray3<f64>>,
        gr: Py<PyArray3<f64>>,
        beta: f64,
        ur: Py<PyArray3<f64>>,
        u_sr: Py<PyArray3<f64>>,
        ur_lr: Py<PyArray3<f64>>,
        uk_lr: Py<PyArray3<f64>>,
        wk: Py<PyArray3<f64>>,
        density: Py<PyArray2<f64>>,
        rgrid: Py<PyArray1<f64>>,
        kgrid: Py<PyArray1<f64>>,
    ) -> Self {
        DataPy {
            temp,
            kt,
            ku,
            amph,
            ns1,
            ns2,
            nsp1,
            nsp2,
            npts,
            radius,
            nlam,
            cr,
            tr,
            hr,
            hk,
            gr,
            beta,
            ur,
            u_sr,
            ur_lr,
            uk_lr,
            wk,
            density,
            rgrid,
            kgrid,
        }
    }
}
