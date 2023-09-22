use std::fmt;
use crate::data::DataRs;
use crate::mdiis::MDIIS;
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyArray3};
use pyo3::prelude::*;

#[derive(FromPyObject, Debug, Clone)]
pub struct Site {
    pub atom_type: String,
    pub params: Vec<f64>,
    pub coords: Vec<f64>,
}

#[derive(FromPyObject, Debug, Clone)]
pub struct Species {
    pub species_name: String,
    pub dens: f64,
    pub ns: usize,
    pub atom_sites: Vec<Site>,
}

#[derive(FromPyObject, Debug, Clone)]
pub struct DataConfig {
    pub temp: f64,
    pub kt: f64,
    pub ku: f64,
    pub amph: f64,
    pub nsv: usize,
    pub nsu: usize,
    pub nspv: usize,
    pub nspu: usize,
    pub npts: usize,
    pub radius: f64,
    pub nlambda: f64,
    pub atoms: Vec<Site>,
    pub solvent_species: Vec<Species>,
    pub solute_species: Vec<Species>,
}

pub enum PotentialKind {
    LennardJones,
    HardSpheres,
    Coulomb,
    NgRenormalisation,
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

pub enum ClosureKind {
    HyperNettedChain,
    KovalenkoHirata,
    PercusYevick,
    PartialSeriesExpansion(i8)
}

impl fmt::Display for ClosureKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
       match self {
           ClosureKind::HyperNettedChain => write!(f, "Hyper-Netted Chain"),
           ClosureKind::KovalenkoHirata => write!(f, "Kovalenko-Hirata"),
           ClosureKind::PercusYevick => write!(f, "Percus-Yevick"),
           ClosureKind::PartialSeriesExpansion(x) => write!(f, "Partial Series Expansion ({} terms)", x),
       }
    }
}

pub enum IntegralEquationKind {
    XRISM,
    DRISM,
}

impl fmt::Display for IntegralEquationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntegralEquationKind::XRISM => write!(f, "Extended RISM"),
            IntegralEquationKind::DRISM => write!(f, "Dielectrically Consistent RISM"),
        }
    }
}

pub struct OperatorConfig {
    pub integral_equation: IntegralEquationKind,
    pub closure: ClosureKind,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RISMDriver {
    data: DataRs,
    solver: MDIIS,
}

#[pymethods]
impl RISMDriver {
    #[new]
    fn new<'py>(
        temp: f64,
        kt: f64,
        amph: f64,
        ns1: usize,
        ns2: usize,
        npts: usize,
        radius: f64,
        nlam: usize,
        ur: PyReadonlyArray3<'py, f64>,
        u_sr: PyReadonlyArray3<'py, f64>,
        ur_lr: PyReadonlyArray3<'py, f64>,
        uk_lr: PyReadonlyArray3<'py, f64>,
        wk: PyReadonlyArray3<'py, f64>,
        density: PyReadonlyArray2<'py, f64>,
        m: usize,
        mdiis_damping: f64,
        picard_damping: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> PyResult<Self> {
        let data = DataRs::new(
            temp,
            kt,
            amph,
            ns1,
            ns2,
            npts,
            radius,
            nlam,
            ur.as_array().to_owned(),
            u_sr.as_array().to_owned(),
            ur_lr.as_array().to_owned(),
            uk_lr.as_array().to_owned(),
            wk.as_array().to_owned(),
            density.as_array().to_owned(),
        );
        let solver = MDIIS::new(m, mdiis_damping, picard_damping, max_iter, tolerance, npts, ns1, ns2);
        Ok(RISMDriver {
            data,
            solver,
        })
    }

    pub fn do_rism(&mut self) {
        println!("{}", ClosureKind::PartialSeriesExpansion(3));
        println!("{}", IntegralEquationKind::DRISM);
        println!("{}", PotentialKind::Coulomb);
        self.solver.solve(&mut self.data);
    }

    #[staticmethod]
    pub fn data_config_build(dataconfig: &PyAny) {
        let data: DataConfig = dataconfig.extract().expect("could not extract");
        println!("{:#?}", data);
    }

    pub fn extract<'py>(
        &'py self,
        py: Python<'py>,
    ) -> PyResult<(
        &PyArray3<f64>,
        &PyArray3<f64>,
        &PyArray3<f64>,
        &PyArray3<f64>,
    )> {
        Ok((
            self.data.cr.clone().into_pyarray(py),
            self.data.tr.clone().into_pyarray(py),
            self.data.hr.clone().into_pyarray(py),
            self.data.hk.clone().into_pyarray(py),
        ))
    }
}

