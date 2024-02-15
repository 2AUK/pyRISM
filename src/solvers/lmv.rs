use crate::data::{configuration::solver::*, core::DataRs};
use crate::iet::operator::Operator;
use crate::solvers::solver::Solver;
use ndarray::Array1;

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

    pub fn tabulate_coefficients(_problem: &mut DataRs) -> Array1<f64> {
        todo!()
    }
}

impl Solver for LMV {
    fn solve(
        &mut self,
        _problem: &mut DataRs,
        _operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        todo!()
    }
}
