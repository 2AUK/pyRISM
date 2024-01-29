use crate::data::{configuration::solver::*, core::DataRs};
use crate::grids::transforms::fourier_bessel_transform_fftw;
use crate::iet::operator::Operator;
use crate::solvers::solver::Solver;
use log::{info, trace};
use ndarray_linalg::Inverse;
use numpy::ndarray::{Array, Array1, Array2, Array3, Array4, Axis, Zip};
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

    fn get_invwc1w(&mut self, problem: &DataRs) -> Array3<f64> {
        let cr = problem.correlations.cr.clone();
        let rho = problem.data_a.borrow().density.clone();
        let b = problem.system.beta;
        let uk_lr = problem.interactions.uk_lr.clone();
        let mut ck = Array::zeros(cr.raw_dim());
        let mut out = Array::zeros(cr.raw_dim());
        let wk = problem.data_b.borrow().wk.clone();
        let r = problem.grid.rgrid.view();
        let k = problem.grid.kgrid.view();
        let dr = problem.grid.dr;
        let rtok = 2.0 * PI * dr;
        let (_, ns2, _) = cr.dim();
        let identity = Array::eye(ns2);
        // Transforming c(r) -> c(k)
        Zip::from(cr.lanes(Axis(0)))
            .and(ck.lanes_mut(Axis(0)))
            .par_for_each(|cr_lane, mut ck_lane| {
                ck_lane.assign(&fourier_bessel_transform_fftw(
                    rtok,
                    &r,
                    &k,
                    &cr_lane.to_owned(),
                ));
            });

        ck = ck - b * uk_lr.to_owned();
        println!("{:?}", ck);
        println!("{:?}", r);
        println!("{:?}", k);

        Zip::from(out.outer_iter_mut())
            .and(wk.outer_iter())
            .and(ck.outer_iter())
            .for_each(|mut out_matrix, wk_matrix, ck_matrix| {
                println!("wc\n{:?}", &wk_matrix.dot(&ck_matrix));
                println!("wcp\n{:?}", &wk_matrix.dot(&ck_matrix.dot(&rho)));
                println!("pwc\n{:?}", &rho.dot(&wk_matrix.dot(&ck_matrix)));
                println!(
                    "1 - wcp\n{:?}",
                    &identity - &wk_matrix.dot(&ck_matrix.dot(&rho))
                );
                let inv1wcp = (&identity - &wk_matrix.dot(&ck_matrix.dot(&rho)))
                    .inv()
                    .expect("Matrix inversion of 1.0 - w * c * rho");
                let result = inv1wcp.dot(&wk_matrix);
                println!("(1-wcp)^-1 * w\n{:?}", result);
                out_matrix.assign(&result);
            });
        out
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

        // let c_prev = problem.correlations.cr.clone();
        // (operator.eq)(problem);
        // let c_a = (operator.closure)(&problem);
        // problem.correlations.cr = self.step_picard(&c_a, &c_prev);
        //
        loop {
            let c_prev = problem.correlations.cr.clone();
            (operator.eq)(problem);
            let c_a = (operator.closure)(&problem);
            let c_next;

            let invwc1w = self.get_invwc1w(&problem);
            println!("{}", invwc1w);
            c_next = c_a;

            problem.correlations.cr = c_next.clone();
            let rmse = conv_rmse(ns1, ns2, npts, problem.grid.dr, &c_next, &c_prev);

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
        }
    }
}

fn compute_rmse(
    ns1: usize,
    _ns2: usize,
    npts: usize,
    curr: &Array3<f64>,
    prev: &Array3<f64>,
) -> f64 {
    (1.0 / ns1 as f64 / npts as f64 * (curr - prev).sum().powf(2.0)).sqrt()
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
