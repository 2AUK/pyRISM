use crate::data::{DataPy, DataRs};
use crate::mdiis::MDIIS;
use numpy::{
    IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray,
};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

pub mod closure;
pub mod data;
pub mod mdiis;
pub mod transforms;
pub mod xrism;
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_helpers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<MDIIS>()?;
    m.add_class::<DataPy>()?;
    m.add_class::<DataRs>()?;

    #[pyfn(m)]
    fn extract_mdiis<'py>(
        py: Python<'py>,
        mdiis: MDIIS,
    ) -> (
        &PyArray3<f64>,
        &PyArray3<f64>,
        &PyArray3<f64>,
        &PyArray3<f64>,
    ) {
        (
            mdiis.data.cr.into_pyarray(py),
            mdiis.data.tr.into_pyarray(py),
            mdiis.data.hr.into_pyarray(py),
            mdiis.data.hk.into_pyarray(py),
        )
    }

    Ok(())
}
