use crate::data::configuration::{
    Configuration,
    {
        operator::OperatorConfig, potential::PotentialConfig, problem::ProblemConfig,
        solver::SolverConfig,
    },
};
use crate::data::{Correlations, Interactions};
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

pub struct Solutions {
    pub config: Configuration,
    pub vv: SolvedData,
    pub uv: Option<SolvedData>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SolvedData {
    pub data_config: ProblemConfig,
    pub solver_config: SolverConfig,
    pub potential_config: PotentialConfig,
    pub operator_config: OperatorConfig,
    pub interactions: Interactions,
    pub correlations: Correlations,
}

impl SolvedData {
    pub fn new(
        data_config: ProblemConfig,
        solver_config: SolverConfig,
        potential_config: PotentialConfig,
        operator_config: OperatorConfig,
        interactions: Interactions,
        correlations: Correlations,
    ) -> Self {
        SolvedData {
            data_config,
            solver_config,
            potential_config,
            operator_config,
            interactions,
            correlations,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyCorrelations {
    #[pyo3(get, set)]
    pub cr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub tr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub hr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub gr: Py<PyArray3<f64>>,
}

impl PyCorrelations {
    pub fn new<'py>(
        cr: Array3<f64>,
        tr: Array3<f64>,
        hr: Array3<f64>,
        gr: Array3<f64>,
        py: Python<'py>,
    ) -> Self {
        PyCorrelations {
            cr: cr.into_pyarray(py).into(),
            tr: tr.into_pyarray(py).into(),
            hr: hr.into_pyarray(py).into(),
            gr: gr.into_pyarray(py).into(),
        }
    }

    pub fn from_correlations<'py>(corr: Correlations, py: Python<'py>) -> Self {
        let gr = 1.0 + &corr.hr;
        Self::new(corr.cr, corr.tr, corr.hr, gr, py)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyInteractions {
    #[pyo3(get, set)]
    pub ur: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub u_sr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub ur_lr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub uk_lr: Py<PyArray3<f64>>,
}

impl PyInteractions {
    pub fn new<'py>(
        ur: Array3<f64>,
        u_sr: Array3<f64>,
        ur_lr: Array3<f64>,
        uk_lr: Array3<f64>,
        py: Python<'py>,
    ) -> Self {
        PyInteractions {
            ur: ur.into_pyarray(py).into(),
            u_sr: u_sr.into_pyarray(py).into(),
            ur_lr: ur_lr.into_pyarray(py).into(),
            uk_lr: uk_lr.into_pyarray(py).into(),
        }
    }

    pub fn from_interactions<'py>(inter: Interactions, py: Python<'py>) -> Self {
        Self::new(inter.ur, inter.u_sr, inter.ur_lr, inter.uk_lr, py)
    }
}
#[pyclass]
#[derive(Clone)]
pub struct PySolvedData {
    #[pyo3(get, set)]
    pub interactions: PyInteractions,
    #[pyo3(get, set)]
    pub correlations: PyCorrelations,
}

#[pyclass]
pub struct PySolution {
    #[pyo3(get, set)]
    pub vv: PySolvedData,
    #[pyo3(get, set)]
    pub uv: Option<PySolvedData>,
}
