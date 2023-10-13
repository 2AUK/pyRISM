use crate::data::{Correlations, Interactions};
use crate::operator::OperatorConfig;
use crate::potential::PotentialConfig;
use crate::solver::SolverConfig;
use numpy::PyArray3;
use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub struct SolvedData {
    pub solver_config: SolverConfig,
    pub potential_config: PotentialConfig,
    pub operator_config: OperatorConfig,
    pub interactions: Interactions,
    pub correlations: Correlations,
}

impl SolvedData {
    pub fn new(
        solver_config: SolverConfig,
        potential_config: PotentialConfig,
        operator_config: OperatorConfig,
        interactions: Interactions,
        correlations: Correlations,
    ) -> Self {
        SolvedData {
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

// #[pymethods]
// impl PyCorrelations {
//     pub fn new(cr: Array3<f64>, tr: Array3<f64>, hr: Array3<f64>, gr: Array3<f 64>) -> Self {
//         PyCorrelations {
//             cr: cr.into
//         }
//     }
// }
//
// pub struct PySolvedData {}
