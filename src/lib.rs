use crate::xrism::xrism_vv_equation;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

pub mod transforms;
pub mod xrism;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_helpers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    #[pyfn(m)]
    fn xrism<'py>(
        py: Python<'py>,
        ns: usize,
        npts: usize,
        r: PyReadonlyArray1<f64>,
        k: PyReadonlyArray1<f64>,
        dr: f64,
        dk: f64,
        cr: PyReadonlyArray3<f64>,
        wk: PyReadonlyArray3<f64>,
        p: PyReadonlyArray2<f64>,
        B: f64,
        uk_lr: PyReadonlyArray3<f64>,
        ur_lr: PyReadonlyArray3<f64>,
    ) -> PyResult<&'py PyTuple> {
        let (hk, tr) = xrism_vv_equation(
            ns,
            npts,
            r.as_array(),
            k.as_array(),
            dr,
            dk,
            cr.as_array(),
            wk.as_array(),
            p.as_array(),
            B,
            uk_lr.as_array(),
            ur_lr.as_array(),
        );
        let elements = vec![hk.to_pyarray(py), tr.to_pyarray(py)];
        let tuple = PyTuple::new(py, elements);
        Ok(tuple)
    }

    Ok(())
}
