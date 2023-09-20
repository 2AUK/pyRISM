use crate::data::DataRs;
use crate::mdiis::MDIIS;
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyArray3};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct RISMController {
    data: DataRs,
    solver: MDIIS,
}

#[pymethods]
impl RISMController {
    #[new]
    fn new<'py>(
        temp: f64,
        kt: f64,
        amph: f64,
        ns1: usize,
        ns2: usize,
        npts: usize,
        radius: f64,
        nlam: usize,
        ur: PyReadonlyArray3<'py, f64>,
        u_sr: PyReadonlyArray3<'py, f64>,
        ur_lr: PyReadonlyArray3<'py, f64>,
        uk_lr: PyReadonlyArray3<'py, f64>,
        wk: PyReadonlyArray3<'py, f64>,
        density: PyReadonlyArray2<'py, f64>,
        m: usize,
        mdiis_damping: f64,
        picard_damping: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> PyResult<Self> {
        let data = DataRs::new(
            temp,
            kt,
            amph,
            ns1,
            ns2,
            npts,
            radius,
            nlam,
            ur.as_array().to_owned(),
            u_sr.as_array().to_owned(),
            ur_lr.as_array().to_owned(),
            uk_lr.as_array().to_owned(),
            wk.as_array().to_owned(),
            density.as_array().to_owned(),
        );
        let solver = MDIIS::new(m, mdiis_damping, picard_damping, max_iter, tolerance, npts, ns1, ns2);
        Ok(RISMController {
            data,
            solver,
        })
    }

    pub fn do_rism(&mut self) {
        self.solver.solve(&mut self.data);
    }

    pub fn extract<'py>(
        &'py self,
        py: Python<'py>,
    ) -> PyResult<(
        &PyArray3<f64>,
        &PyArray3<f64>,
        &PyArray3<f64>,
        &PyArray3<f64>,
    )> {
        Ok((
            self.data.cr.clone().into_pyarray(py),
            self.data.tr.clone().into_pyarray(py),
            self.data.hr.clone().into_pyarray(py),
            self.data.hk.clone().into_pyarray(py),
        ))
    }
}

