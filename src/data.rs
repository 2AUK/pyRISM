use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

#[pyclass]
pub struct Data {
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
impl Data {
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
        Data {
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
