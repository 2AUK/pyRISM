use crate::{
    data::core::{Correlations, Interactions},
    thermodynamics::thermo::{Densities, SFEs, Thermodynamics},
};
use ndarray::{Array1, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray3};
use pyo3::prelude::*;
use std::f64::consts::PI;

#[pyclass]
#[derive(Clone)]
pub struct PyGrid {
    #[pyo3(get, set)]
    pub rgrid: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    pub kgrid: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    pub npts: usize,
    #[pyo3(get, set)]
    pub radius: f64,
    #[pyo3(get, set)]
    pub dr: f64,
    #[pyo3(get, set)]
    pub dk: f64,
}

impl PyGrid {
    pub fn new<'py>(npts: usize, radius: f64, py: Python<'py>) -> Self {
        let dr = radius / npts as f64;
        let dk = 2.0 * PI / (2.0 * npts as f64 * dr);
        let rgrid = Array1::range(0.5, npts as f64, 1.0) * dr;
        let kgrid = Array1::range(0.5, npts as f64, 1.0) * dk;
        PyGrid {
            rgrid: rgrid.into_pyarray(py).into(),
            kgrid: kgrid.into_pyarray(py).into(),
            npts,
            radius,
            dr,
            dk,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySFEs {
    #[pyo3(get, set)]
    pub hypernettedchain: f64,
    #[pyo3(get, set)]
    pub kovalenko_hirata: f64,
    #[pyo3(get, set)]
    pub gaussian_fluctuations: f64,
    #[pyo3(get, set)]
    pub partial_wave: f64,
    #[pyo3(get, set)]
    pub pc_plus: f64,
}

impl PySFEs {
    pub fn from_sfes(sfes: SFEs) -> Self {
        PySFEs {
            hypernettedchain: sfes.hypernettedchain,
            kovalenko_hirata: sfes.kovalenko_hirata,
            gaussian_fluctuations: sfes.gaussian_fluctuations,
            partial_wave: sfes.partial_wave,
            pc_plus: sfes.partial_wave,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDensities {
    #[pyo3(get, set)]
    pub hypernettedchain: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    pub kovalenko_hirata: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    pub gaussian_fluctuations: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    pub partial_wave: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    pub partial_molar_volume: Py<PyArray1<f64>>,
}

impl PyDensities {
    pub fn new<'py>(
        hnc: Array1<f64>,
        kh: Array1<f64>,
        gf: Array1<f64>,
        pw: Array1<f64>,
        pmv: Array1<f64>,
        py: Python<'py>,
    ) -> Self {
        PyDensities {
            hypernettedchain: hnc.into_pyarray(py).into(),
            kovalenko_hirata: kh.into_pyarray(py).into(),
            gaussian_fluctuations: gf.into_pyarray(py).into(),
            partial_wave: pw.into_pyarray(py).into(),
            partial_molar_volume: pmv.into_pyarray(py).into(),
        }
    }

    pub fn from_densities<'py>(densities: Densities, py: Python<'py>) -> Self {
        Self::new(
            densities.hypernettedchain,
            densities.kovalenko_hirata,
            densities.gaussian_fluctuations,
            densities.partial_wave,
            densities.partial_molar_volume,
            py,
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyThermodynamics {
    #[pyo3(get, set)]
    pub total_density: f64,
    #[pyo3(get, set)]
    pub temperature: f64,
    #[pyo3(get, set)]
    pub isothermal_compressibility: f64,
    #[pyo3(get, set)]
    pub sfe: Option<PySFEs>,
    #[pyo3(get, set)]
    pub sfed: Option<PyDensities>,
    pub molecular_kb_pmv: Option<f64>,
    #[pyo3(get, set)]
    pub rism_kb_pmv: Option<f64>,
    #[pyo3(get, set)]
    pub pressure: Option<f64>,
}

impl PyThermodynamics {
    pub fn from_thermodynamics<'py>(thermodynamics: Thermodynamics, py: Python<'py>) -> Self {
        let (sfe, sfed) = match thermodynamics.sfe {
            Some(sfe) => {
                let sfe = Some(PySFEs::from_sfes(sfe));
                let sfed = Some(PyDensities::from_densities(
                    thermodynamics.sfed.unwrap(),
                    py,
                ));
                (sfe, sfed)
            }
            None => (None, None),
        };

        PyThermodynamics {
            total_density: thermodynamics.total_density,
            temperature: thermodynamics.temperature,
            isothermal_compressibility: thermodynamics.isothermal_compressibility,
            sfe,
            sfed,
            molecular_kb_pmv: thermodynamics.molecular_kb_pmv,
            rism_kb_pmv: thermodynamics.rism_kb_pmv,
            pressure: thermodynamics.pressure,
        }
    }
}

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

#[pyclass]
pub struct PyJobDetails {}