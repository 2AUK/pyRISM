use numpy::ndarray::{Array, Array1, Array2, Array3, Axis, Zip, ArrayView1, ArrayView3, ArrayView2};
use ndarray_linalg::Inverse;
use pyo3::prelude::*;
use numpy::PyArray3;


#[pyclass]
struct MDIIS {
    pub cr: Array3<f64>,
    pub tr: Array3<f64>,
    pub hk: Array3<f64>
}
