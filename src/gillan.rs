use crate::{
    data::DataRs,
    operator::Operator,
    solver::{Solver, SolverError, SolverSettings, SolverSuccess},
};
use log::trace;
use ndarray::{Array, Array1, Array3};

#[derive(Clone, Debug)]
pub struct Gillan {
    // input parameters for solver
    pub nbasis: usize,
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,
}
impl Gillan {
    pub fn new(settings: &SolverSettings) -> Self {
        let gillan_settings = settings
            .clone()
            .gillan_settings
            .expect("MDIIS settings not found");
        Gillan {
            nbasis: gillan_settings.nbasis,
            picard_damping: settings.picard_damping,
            max_iter: settings.max_iter,
            tolerance: settings.tolerance,
        }
    }

    fn step_picard(&mut self, curr: &Array3<f64>, prev: &Array3<f64>) -> Array3<f64> {
        // calculate difference between current and previous solutions from RISM equation
        let diff = curr.clone() - prev.clone();

        // return Picard iteration step
        prev + self.picard_damping * diff
    }

    pub fn step_gillan(&mut self, curr: &Array3<f64>, prev: &Array3<f64>) -> Array1<f64> {
        Array::from_iter(curr.clone().into_iter())
    }

    pub fn compute_basis_functions(&mut self) {
        todo!()
    }
}

impl Solver for Gillan {
    fn solve(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        let shape = problem.correlations.cr.dim();
        let mut i = 0;

        let result = loop {
            let c_prev = problem.correlations.cr.clone();
            (operator.eq)(problem);
            let c_a = (operator.closure)(&problem);
            let c_next;
            c_next = self
                .step_gillan(&c_a, &c_prev)
                .into_shape(shape)
                .expect("could not reshape array into original shape");
            problem.correlations.cr = c_next.clone();
            let rmse = conv_rmse(&c_prev);

            trace!("Iteration: {} Convergence RMSE: {:.6E}", i, rmse);

            if rmse <= self.tolerance {
                break Ok(SolverSuccess(i, rmse));
            }

            if rmse == std::f64::NAN || rmse == std::f64::INFINITY {
                break Err(SolverError::ConvergenceError(i));
            }

            i += 1;

            if i == self.max_iter {
                break Err(SolverError::MaxIterationError(i));
            }
        };
        result
    }
}

fn conv_rmse(res: &Array3<f64>) -> f64 {
    let denom = 1.0 / res.len() as f64;
    (res.mapv(|x| x.powf(2.0)).sum() * denom).sqrt()
}
