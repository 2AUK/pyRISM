use crate::{
    data::core::{Correlations, Interactions},
    grids::radial_grid::Grid,
};
use ndarray::{Array1, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray3};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PyGrid {
    pub rgrid: Py<PyArray1<f64>>,
    pub kgrid: Py<PyArray1<f64>>,
    pub npts: usize,
    pub radius: f64,
    pub dr: f64,
    pub dk: f64,
}

impl PyGrid {
    pub fn new<'py>(
        rgrid: Array1<f64>,
        kgrid: Array1<f64>,
        npts: usize,
        radius: f64,
        dr: f64,
        dk: f64,
        py: Python<'py>,
    ) -> Self {
        PyGrid {
            rgrid: rgrid.into_pyarray(py).into(),
            kgrid: kgrid.into_pyarray(py).into(),
            npts,
            radius,
            dr,
            dk,
        }
    }

    pub fn from_grid<'py>(grid: Grid, py: Python<'py>) -> Self {
        Self::new(
            grid.rgrid,
            grid.kgrid,
            grid.npts,
            grid.radius,
            grid.dr,
            grid.dk,
            py,
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySFEs {}

#[pyclass]
#[derive(Clone)]
pub struct PyDensities {}

#[pyclass]
#[derive(Clone)]
pub struct PyThermodynamics {}

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
