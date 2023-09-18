use ndarray::{Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use std::f64::consts::PI;

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
    fn new(npts: usize, radius: f64) -> Self {
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

#[pyclass]
#[derive(Clone, Debug)]
pub struct DataRs {
    pub temp: f64,
    pub kt: f64,
    pub amph: f64,
    pub ns1: usize,
    pub ns2: usize,
    pub nlam: usize,

    pub grid: Grid,

    pub cr: Array3<f64>,
    pub tr: Array3<f64>,
    pub hr: Array3<f64>,
    pub hk: Array3<f64>,

    pub beta: f64,

    // This set of arrays can be taken directly from python
    pub ur: Array3<f64>,
    pub u_sr: Array3<f64>,
    pub ur_lr: Array3<f64>,
    pub uk_lr: Array3<f64>,
    pub wk: Array3<f64>,
    pub density: Array2<f64>,
}

#[pymethods]
impl DataRs {
    #[new]
    fn new(
        temp: f64,
        kt: f64,
        amph: f64,
        ns1: usize,
        ns2: usize,
        npts: usize,
        radius: f64,
        nlam: usize,
        ur: PyReadonlyArray3<f64>,
        u_sr: PyReadonlyArray3<f64>,
        ur_lr: PyReadonlyArray3<f64>,
        uk_lr: PyReadonlyArray3<f64>,
        wk: PyReadonlyArray3<f64>,
        density: PyReadonlyArray2<f64>,
    ) -> Self {
        let shape = (npts, ns1, ns2);
        let grid = Grid::new(npts, radius);
        DataRs {
            temp,
            kt,
            amph,
            ns1,
            ns2,
            grid,
            nlam,
            cr: Array3::zeros(shape),
            tr: Array3::zeros(shape),
            hr: Array3::zeros(shape),
            hk: Array3::zeros(shape),
            beta: 1.0 / temp / kt,
            ur: ur.as_array().to_owned(),
            u_sr: u_sr.as_array().to_owned(),
            ur_lr: ur_lr.as_array().to_owned(),
            uk_lr: uk_lr.as_array().to_owned(),
            wk: wk.as_array().to_owned(),
            density: density.as_array().to_owned(),
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
