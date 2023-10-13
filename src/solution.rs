use crate::data::{Correlations, DataConfig, Interactions};
use crate::operator::OperatorConfig;
use crate::potential::PotentialConfig;
use crate::solver::SolverConfig;
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SolvedData {
    pub data_config: DataConfig,
    pub solver_config: SolverConfig,
    pub potential_config: PotentialConfig,
    pub operator_config: OperatorConfig,
    pub interactions: Interactions,
    pub correlations: Correlations,
}

impl SolvedData {
    pub fn new(
        data_config: DataConfig,
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
}

#[pyclass]
pub struct PyInteractions {}

pub struct PySolvedData {}
