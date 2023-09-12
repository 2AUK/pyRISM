use crate::closure::hyper_netted_chain;
use crate::mdiis::MDIIS;
use crate::data::Data;
use crate::xrism::xrism_vv_equation;
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

    #[pyfn(m)]
    fn hnc<'py>(
        py: Python<'py>,
        b: f64,
        u: PyReadonlyArray3<f64>,
        t: PyReadonlyArray3<f64>,
    ) -> PyResult<&'py PyArray3<f64>> {
        Ok(hyper_netted_chain(b, u.as_array(), t.as_array()).to_pyarray(py))
    }

    m.add_class::<MDIIS>()?;
    m.add_class::<Data>()?;

    Ok(())
}
