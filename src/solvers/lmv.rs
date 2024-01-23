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

    fn get_cjk(&mut self, costab: Array2<f64>, der: Array3<f64>) -> Array4<f64> {
        let shape = der.dim();
        let (npts, ns1, ns2) = shape;
        let mut dp: Array1<f64> = Array::zeros(3 * self.nbasis);
        let mut cjk: Array4<f64> = Array::zeros((ns1, ns2, self.nbasis, self.nbasis));

        for i in 0..ns1 {
            for j in 0..ns2 {
                println!("ns idx: {},{}", i, j);
                for m in 1..(3 * self.nbasis) - 1 {
                    dp[m] = 0.0;
                    for l in 0..npts {
                        println!("point: {}, basis: {}, before: {}", l, m, dp[[m]]);
                        println!(
                            "der: {}, costab: {}, der * costab: {}",
                            der[[l, i, j]],
                            costab[[m, l]],
                            der[[l, i, j]] * costab[[m, l]]
                        );
                        dp[[m]] += der[[l, i, j]] * costab[[m, l]];
                        println!("point: {}, basis: {}, after: {}", l, m, dp[[m]]);
                    }
                    dp[m] = dp[m] / npts as f64;
                }
                for m in 0..(3 * self.nbasis) {
                    println!("dp[{}] = {}", m, dp[m]);
                }

                for m in 0..self.nbasis {
                    for k in 0..self.nbasis {
                        let kpm_m = k + m + self.nbasis;
                        let kmm_m = (k as isize - m as isize + self.nbasis as isize) as usize;
                        println!("m: {}, k: {}, k-m+M: {}, k+m+M: {}", m, k, kpm_m, kmm_m,);
                        cjk[[i, j, m, k]] = dp[kmm_m] - dp[kpm_m];
                        println!(
                            "c_{}{}[{}][{}] = {} - {} = {}",
                            i,
                            j,
                            m,
                            k,
                            dp[kmm_m],
                            dp[kpm_m],
                            cjk[[i, j, m, k]]
                        );
                    }
                }
            }
        }
        cjk
    }

    fn get_jacobian(&mut self, invwc1w: Array3<f64>) -> Array3<f64> {
        todo!()
    }

    fn lmv_update(&mut self, der: Array3<f64>, invwc1w: Array3<f64>) {
        self.get_cjk(self.cos_table.clone().unwrap(), der);
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
                            .cos();
                    println!("outarr[{}][{}] = {}", j, i, out_arr[[j, i]]);
                }
            }
            Some(out_arr)
        };

        Ok(SolverSuccess(1, 0.1))
    }
}
