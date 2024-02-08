use crate::data::{configuration::solver::*, core::DataRs};
use crate::grids::transforms::fourier_bessel_transform_fftw;
use crate::iet::operator::Operator;
use crate::solvers::solver::Solver;
use itertools::Itertools;
use log::{debug, info, trace};
use ndarray_linalg::{Inverse, Solve};
use numpy::ndarray::{
    s, Array, Array1, Array2, Array3, Array4, ArrayView2, Axis, IntoNdProducer, Slice, Zip,
};
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

    pub fn tabulate_coefficients(&self, derivative: &Array3<f64>) -> Array4<f64> {
        let (npts, ns1, ns2) = derivative.dim();
        let shape = (self.nbasis, self.nbasis, ns1, ns2);
        let mut out_arr = Array::zeros(shape);

        // Zip::indexed(derivative.lanes(Axis(0))).for_each(|idx, der_mat| {
        //     let (a, b) = idx;
        //     for ((l, val), v) in der_mat.indexed_iter().zip(0..self.nbasis - 1) {
        //         let n = v + 1;
        //         out_arr[[a, b, v, v]] +=
        //             val * (1.0 - (PI * l as f64 * (v + v) as f64 / npts as f64).cos());
        //         out_arr[[a, b, v, n]] += val
        //             * ((PI * l as f64 / npts as f64).cos()
        //                 - (PI * l as f64 * (v + v) as f64 / npts as f64).cos());
        //     }
        // });

        Zip::indexed(derivative.lanes(Axis(0))).for_each(|idx, der_mat| {
            let (a, b) = idx;
            // let nbasis_idx = (0..self.nbasis)
            //     .into_iter()
            //     .cartesian_product((0..self.nbasis).into_iter());
            for (l, cr) in der_mat.indexed_iter() {
                for v in 0..self.nbasis {
                    for n in 0..self.nbasis {
                        //println!("{}, {}, {}, {}", l, cr, v, n);
                        out_arr[[v, n, a, b]] += cr
                            * ((PI * l as f64 * (n as f64 - v as f64) / npts as f64).cos()
                                - (PI * l as f64 * (n as f64 + v as f64) / npts as f64).cos());
                    }
                }
            }

            // for (v, n) in nbasis_idx {
            //     out_arr[[a, b, n, v]] = out_arr[[a, b, v, n]];
            // }
        });

        out_arr = out_arr / npts as f64;
        // println!("{}", out_arr);

        out_arr
    }

    pub fn jacobian(
        &self,
        cjk: &Array4<f64>,
        tk_delta: &Array3<f64>,
        invwc1w: &Array3<f64>,
    ) -> (Array1<f64>, Array2<f64>) {
        let (npts, ns1, ns2) = tk_delta.dim();
        let mut matrix = Array::zeros((self.nbasis, self.nbasis, ns1, ns2));
        Zip::from(invwc1w.outer_iter()).for_each(|mat| {
            for v in 0..self.nbasis {
                for n in 0..self.nbasis {
                    let identity: Array2<f64> = Array::eye(ns1);
                    let cjk_ab = cjk.slice(s![v, n, .., ..]).to_owned();
                    let mut matrix_ab = matrix.slice_mut(s![v, n, .., ..]);
                    println!("{}", cjk_ab);
                    println!("{}", mat);
                    matrix_ab.assign(&(identity + &cjk_ab - mat.dot(&cjk_ab.dot(&mat))));
                }
            }
        });
        let vector = {
            let tk_delta_m = tk_delta.slice(s![0..self.nbasis, .., ..]).to_owned();
            Array::from_iter(tk_delta_m.clone())
        };

        (
            vector,
            matrix
                .into_shape((ns1 * self.nbasis, ns2 * self.nbasis))
                .expect("reshaping 4D layout to 2D Jacobian matrix"),
        )
    }
}

impl Solver for LMV {
    fn solve(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        // Initialise iteration counter
        let mut i = 0;
        // Want to iterate the cycle:
        // t(r) -> c'(r) -> t'(k) -> t'(r)
        // We cycle the initial c(r) = 0 guess once with the RISM equation

        // Generate initial guess for t(r)
        (operator.eq)(problem);

        let result = loop {
            // Store previous t(r)
            let tr_prev = problem.correlations.tr.clone();

            // Store previous t(k)
            let tk_prev = problem.correlations.tk.clone();

            // Compute c'(r) from closure
            problem.correlations.cr = (operator.closure)(problem);

            // Compute new t(r)
            (operator.eq)(problem);

            // Store current t(k)
            let tk_curr = problem.correlations.tk.clone();

            // Compute difference between the current and previous t(k)
            let delta_tk = tk_curr - tk_prev;

            // Compute dc'(r)/dt(r)
            let dcrtr = (operator.closure_der)(problem);

            // Compute coefficients from derivative
            let cjk = self.tabulate_coefficients(&dcrtr);

            let (vec, mat) = self.jacobian(&cjk, &delta_tk, &problem.correlations.invwc1wk);

            // Start innner convergence loop
            loop {
                // Compute
                todo!()
            }

            let rms = 1E20;
            if rms <= self.tolerance {
                problem.correlations.cr = (operator.closure)(problem);
                break Ok(SolverSuccess(i, rms));
            }

            if rms == std::f64::NAN || rms == std::f64::INFINITY {
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
