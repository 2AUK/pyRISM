use crate::data::{configuration::solver::*, core::DataRs};
use crate::iet::operator::Operator;
use crate::solvers::solver::Solver;
use log::{info, trace};
use ndarray_linalg::Solve;
use numpy::ndarray::{Array, Array1, Array2, Array3, Array4};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct LMV {
    pub nbasis: usize,
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,
    pub cos_table: Option<Array2<f64>>,
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
            cos_table: None,
        }
    }

    fn step_picard(&mut self, curr: &Array3<f64>, prev: &Array3<f64>) -> Array3<f64> {
        let diff = curr.clone() - prev.clone();
        // return Picard iteration step
        prev + self.picard_damping * diff
    }

    fn get_cjk(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
        costab: Array2<f64>,
    ) -> Array4<f64> {
        let shape = problem.correlations.cr.dim();
        let (npts, ns1, ns2) = shape;
        let mut dp: Array1<f64> = Array::zeros(3 * self.nbasis);
        let der = (operator.closure_der)(&problem);

        for i in 0..ns1 {
            for j in 0..ns2 {
                for m in 1..(3 * self.nbasis) - 1 {
                    for l in 0..npts {
                        dp[[m]] += der[[l, i, j]] * costab[[m, l]];
                    }
                    dp[m] = dp[m] / npts as f64;
                }
            }
        }
        println!("{}", dp);
        Array::zeros((ns1, ns2, self.nbasis, self.nbasis))
    }
}

impl Solver for LMV {
    fn solve(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        let shape = problem.correlations.cr.dim();
        let (npts, ns1, ns2) = shape;
        let mut i = 0;

        // tabulate cos_table
        self.cos_table = {
            let mut out_arr = Array::zeros((3 * self.nbasis, npts));
            for j in 0..3 * self.nbasis {
                for i in 0..npts {
                    out_arr[[j, i]] =
                        (PI * (i as f64 * 2.0 + 1.0) * (j as f64 - self.nbasis as f64)
                            / npts as f64
                            / 2.0)
                            .cos()
                }
            }
            println!("{:?}", out_arr);
            Some(out_arr)
        };

        self.get_cjk(problem, operator, self.cos_table.clone().unwrap());
        Ok(SolverSuccess(1, 0.1))
    }
}
