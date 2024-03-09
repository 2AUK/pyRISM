use crate::data::{configuration::solver::*, core::DataRs};
use crate::iet::operator::Operator;
use crate::solvers::solver::Solver;
use log::{info, trace};
use ndarray_linalg::Solve;
use numpy::ndarray::{Array, Array1, Array3};
use std::collections::VecDeque;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Ng {
    // input parameters for solver
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,

    // Vectors containing 3 previous solutions
    pub fr: VecDeque<Array1<f64>>,
    pub gr: VecDeque<Array1<f64>>,
}

impl Ng {
    pub fn new(settings: &SolverSettings) -> Self {
        Ng {
            picard_damping: settings.picard_damping,
            max_iter: settings.max_iter,
            tolerance: settings.tolerance,
            fr: VecDeque::new(),
            gr: VecDeque::new(),
        }
    }

    fn step_picard(&mut self, curr: &Array3<f64>, prev: &Array3<f64>) -> Array3<f64> {
        // calculate difference between current and previous solutions from RISM equation
        let diff = curr.clone() - prev.clone();

<<<<<<< HEAD:src/solvers/ng.rs
        self.fr.push_back(Array::from_iter(prev.clone()));

        self.gr.push_back(Array::from_iter(curr.clone()));
=======
        self.fr.push_back(Array::from_iter(curr.clone()));

        self.gr.push_back(Array::from_iter(prev.clone()));
>>>>>>> fdef8b3db084711ac2eb3aff1a01f40a7e042ea9:src/ng.rs
        // return Picard iteration step
        prev + self.picard_damping * diff
    }

    fn step_ng(&mut self, curr: &Array3<f64>, prev: &Array3<f64>) -> Array1<f64> {
        let mut a = Array::zeros((2, 2));
        let mut b = Array::zeros(2);
        let n = 2;
        let d2 = self.gr[n - 2].clone() - self.fr[n - 2].clone();
        let d1 = self.gr[n - 1].clone() - self.fr[n - 1].clone();
        let d0 = self.gr[n].clone() - self.fr[n].clone();

        let dn = d0.clone();
        let d01 = &d0 - &d1;
        let d02 = &d0 - &d2;

        a[[0, 0]] = d01.dot(&d01);
        a[[0, 1]] = d01.dot(&d02);
        a[[1, 0]] = d01.dot(&d02);
        a[[1, 1]] = d02.dot(&d02);

        b[[0]] = dn.dot(&d01);
        b[[1]] = dn.dot(&d02);

        let c = a.solve_into(b).expect("solved coefficients for Ng solver");
<<<<<<< HEAD:src/solvers/ng.rs
        let out = (1.0 - c[[0]] - c[[1]]) * Array::from_iter(self.gr[n].clone())
            + c[0] * Array::from_iter(self.gr[n - 1].clone())
            + c[1] * Array::from_iter(self.gr[n - 2].clone());

        self.fr.push_back(Array::from_iter(prev.clone()));
        self.gr.push_back(Array::from_iter(curr.clone()));
=======
        let out = (1.0 - c[[0]] - c[[1]]) * Array::from_iter(self.gr[2].clone())
            + c[0] * Array::from_iter(self.gr[1].clone())
            + c[1] * Array::from_iter(self.gr[0].clone());

        self.fr.push_back(Array::from_iter(curr.clone()));
        self.gr.push_back(Array::from_iter(prev.clone()));
>>>>>>> fdef8b3db084711ac2eb3aff1a01f40a7e042ea9:src/ng.rs
        self.gr.pop_front();
        self.fr.pop_front();

        out
    }
}

impl Solver for Ng {
    fn solve(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        let timer = Instant::now();
        info! {"Solving RISM equation"};
        let shape = problem.correlations.cr.dim();
        let (npts, ns1, ns2) = shape;
        let mut i = 0;
        self.fr.clear();
        self.gr.clear();

        loop {
            //println!("Iteration: {}", i);
            let c_prev = problem.correlations.cr.clone();
            (operator.eq)(problem);
            let c_a = (operator.closure)(problem);
            let c_next = if i < 3 {
                self.step_picard(&c_a, &c_prev)
            } else {
                self.step_ng(&c_a, &c_prev)
                    .into_shape(c_a.raw_dim())
                    .expect("reshaping Ng step result to original 3D array")
            };
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
