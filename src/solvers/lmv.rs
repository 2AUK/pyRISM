use crate::data::{configuration::solver::*, core::DataRs};
use crate::grids::transforms::fourier_bessel_transform_fftw;
use crate::iet::operator::Operator;
use crate::solvers::solver::Solver;
use log::{debug, info, trace};
use ndarray_linalg::{Inverse, Solve};
use numpy::ndarray::{Array, Array1, Array2, Array3, Array4, Axis, IntoNdProducer, Zip};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct LMV {
    pub nbasis: usize,
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,
}

impl LMV {
    pub fn new(settings: &SolverSettings) -> Self {
        let lmv_settings = settings
            .clone()
            .gillan_settings
            .expect("LMV settings not found");
        LMV {
            nbasis: lmv_settings.nbasis,
            picard_damping: settings.picard_damping,
            max_iter: settings.max_iter,
            tolerance: settings.tolerance,
        }
    }

    pub fn tabulate_coefficients(problem: &mut DataRs) -> Array1<f64> {
        todo!()
    }
}

impl Solver for LMV {
    fn solve(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        todo!()
    }
}
