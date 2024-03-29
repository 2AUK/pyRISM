use crate::data::{configuration::solver::*, core::DataRs};
use crate::iet::operator::Operator;
use crate::solvers::solver::Solver;
use log::{info, trace};
use numpy::ndarray::Array3;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Picard {
    // input parameters for solver
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,
}

impl Picard {
    pub fn new(settings: &SolverSettings) -> Self {
        Picard {
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
}

impl Solver for Picard {
    fn solve(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        info! {"Solving RISM equation"};
        let shape = problem.correlations.cr.dim();
        let (npts, ns1, ns2) = shape;
        let mut i = 0;
        let timer = Instant::now();

        loop {
            //println!("Iteration: {}", i);
            let c_prev = problem.correlations.cr.clone();
            (operator.eq)(problem);
            let c_a = (operator.closure)(problem);
            let c_next = self.step_picard(&c_a, &c_prev);
            problem.correlations.cr = c_next.clone();
            let rmse = conv_rmse(ns1, ns2, npts, problem.grid.dr, &c_next, &c_prev);

            trace!("Iteration: {} Convergence RMSE: {:.6E}", i, rmse);

            if rmse <= self.tolerance {
                let elapsed = timer.elapsed();
                break Ok(SolverSuccess(i, rmse, elapsed.as_secs_f64()));
            }

            if rmse == std::f64::NAN || rmse == std::f64::INFINITY {
                break Err(SolverError::ConvergenceError(i));
            }

            i += 1;

            if i == self.max_iter {
                break Err(SolverError::MaxIterationError(i));
            }
        }
    }
}

fn conv_rmse(
    ns1: usize,
    ns2: usize,
    npts: usize,
    dr: f64,
    curr: &Array3<f64>,
    prev: &Array3<f64>,
) -> f64 {
    let denom = 1.0 / ns1 as f64 / ns2 as f64 / npts as f64;
    (dr * (curr - prev).mapv(|x| x.powf(2.0)).sum() * denom).sqrt()
}
