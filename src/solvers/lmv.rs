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
        k: &Array1<f64>,
        invwc1w: &Array3<f64>,
    ) -> (Array1<f64>, Array2<f64>) {
        let (npts, ns1, ns2) = tk_delta.dim();
        let basis_size = ns1 * ns2 * self.nbasis;
        let mut matrix = Array::zeros((basis_size, basis_size));
        let mut vector = Array::zeros(basis_size);
        // Zip::from(invwc1w.outer_iter()).for_each(|mat| {
        //     for v in 0..self.nbasis {
        //         let mut vector_ab = vector.slice_mut(s![v, .., ..]);
        //         let tk_delta_ab = tk_delta.slice(s![v, .., ..]);
        //         vector_ab.assign(&tk_delta_ab);
        //         for n in 0..self.nbasis {
        //             let identity: Array2<f64> = Array::eye(ns1);
        //             let cjk_ab = cjk.slice(s![v, n, .., ..]).to_owned();
        //             let mut matrix_ab = matrix.slice_mut(s![v, .., .., n, .., ..]);
        //             println!("{}", cjk_ab);
        //             println!("{}", mat);
        //             matrix_ab.assign(&(identity + &cjk_ab - mat.dot(&cjk_ab.dot(&mat))));
        //         }
        //     }
        // });
        for v in 0..self.nbasis {
            for a in 0..ns1 {
                for b in 0..ns2 {
                    vector[[v * ns1 * ns2 + a * ns2 + b]] = k[[v]] * tk_delta[[v, a, b]];
                    for n in 0..self.nbasis {
                        for c in 0..ns1 {
                            for d in 0..ns2 {
                                let kron_ac = {
                                    if a == c {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                };

                                let kron_bd = {
                                    if b == d {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                };

                                let _kron_vn = {
                                    if v == n {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                };

                                let kron_acbd = kron_ac * kron_bd;

                                matrix
                                    [[v * ns1 * ns2 + a * ns2 + b, n * ns1 * ns2 + c * ns2 + d]] =
                                    kron_acbd * (_kron_vn + cjk[[v, n, a, b]])
                                        - invwc1w[[v, a, c]]
                                            * invwc1w[[n, d, b]]
                                            * cjk[[v, n, c, d]];
                            }
                        }
                    }
                }
            }
        }
        let vector = Array::from_iter(vector);

        (
            vector,
            matrix
                .into_shape((basis_size, basis_size))
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
        let mut oc_counter = 0;

        // Store original shape
        let (npts, ns1, ns2) = problem.correlations.cr.dim();

        // Want to iterate the cycle:
        // t(r) -> c'(r) -> t'(k) -> t'(r)
        // We cycle the initial c(r) = 0 guess once with the RISM equation

        // Generate initial guesses t(r)
        (operator.eq)(problem);

        let result = loop {
            // Store previous t(r)
            let tr_prev = problem.correlations.tr.clone();

            // Compute c'(r) from closure
            problem.correlations.cr = (operator.closure)(problem);

            // Compute new t(r)
            (operator.eq)(problem);

            // Initialise inner cycle iteration counter
            let mut ic_counter = 0;

            // Store t^0(k)
            let tk_0 = problem.correlations.tk.clone();

            // Store c^0(k)
            let ck_0 = problem.grid.nd_fbt_r2k(&problem.correlations.cr.clone());

            // Store t_ic(k) (from closure) - initial guess is just t^0(k)
            let mut tk_ic: Array3<f64> = Array::zeros(problem.correlations.tk.raw_dim());

            // Compute dc'(r)/dt(r)
            let dcrtr = (operator.closure_der)(problem);

            // Compute coefficients from derivative
            let cjk = self.tabulate_coefficients(&dcrtr);

            // Store previous RMS IC (initial value arbitrarly large so comparison fails
            // successfully)
            let mut prev_rms = 1e20;

            // Start innner convergence loop
            // tk_ic is the variable being changed per step
            // computed via delta_tk = tk_ic - tk_0.
            // tk_0 remains fixed per IC loop, changes with every OC step.
            let tk_ic_new = loop {
                // Compute \Delta t(k) = t_{ic}(k) - t^0(k) for coefficient step
                let tk_delta = &tk_ic - &tk_0;

                // Compute c(k) from coefficients
                let mut summed_term = Array::zeros((npts, ns1, ns2));
                for l in 0..npts {
                    for a in 0..ns1 {
                        for b in 0..ns2 {
                            if l < self.nbasis {
                                let cjk_abj = cjk.slice(s![.., l, a, b]).sum();
                                summed_term[[l, a, b]] = cjk_abj * tk_delta[[l, a, b]];
                            }
                        }
                    }
                }

                // println!("{}", sum);

                problem.correlations.cr = problem.grid.nd_fbt_k2r(&(&ck_0 + summed_term));

                // Compute intermediate t(k) for Jacobian and new M matrix
                (operator.eq)(problem);

                // Store t(k) for Jacobian
                let tk_jac = problem.correlations.tk.clone();

                // Compute \Delta t(k) = t_{ic}(k) - t^0(k) for Jacobian step
                let tk_delta = &tk_jac - &tk_ic;

                // Compute Jacobian and vector
                let (vec, mat) = self.jacobian(
                    &cjk,
                    &tk_delta,
                    &problem.grid.kgrid,
                    &problem.correlations.invwc1wk,
                );

                // println!("{:+.1e}", mat);

                // Perform linear solve for new \Delta t(k)
                let new_delta_tk_flat = mat
                    .solve_into(vec)
                    .expect("linear solve in NR step for new delta t(k)");

                // Renew tk_ic
                let mut delta_tk_sum = 0.0;
                let mut tk_ic_sum = 0.0;
                for v in 0..self.nbasis {
                    for a in 0..ns1 {
                        for b in 0..ns2 {
                            delta_tk_sum += (new_delta_tk_flat[[v * ns1 * ns2 + a * ns2 + b]]
                                / problem.grid.kgrid[[v]])
                            .powf(2.0);
                            tk_ic[[v, a, b]] = tk_ic[[v, a, b]]
                                + self.picard_damping
                                    * (new_delta_tk_flat[[v * ns1 * ns2 + a * ns2 + b]]
                                        / problem.grid.kgrid[[v]]);
                            tk_ic_sum += tk_ic[[v, a, b]].powf(2.0);
                        }
                    }
                }

                // Compute RMS
                let rms = (delta_tk_sum / tk_ic_sum).sqrt() / (self.nbasis * ns1 * ns2) as f64;
                if (rms - prev_rms).abs() < 1e-3 {
                    break tk_ic;
                }
                prev_rms = rms;
                trace!("IC Iteration: {} Convergence RMSE: {:.6E}", ic_counter, rms);

                if rms < 1e-3 {
                    break tk_ic;
                }

                ic_counter += 1;
            };

            let tr_curr = problem.grid.nd_fbt_k2r(&(tk_0 + tk_ic_new));

            let tr_new = tr_curr;

            let rms =
                (&tr_new - &tr_prev).mapv(|x| x.powf(2.0)).sum().sqrt() / (npts * ns1 * ns2) as f64;

            problem.correlations.tr = tr_new;

            trace!("OC Iteration: {} Convergence RMSE: {:.6E}", oc_counter, rms);

            if rms <= self.tolerance {
                problem.correlations.cr = (operator.closure)(problem);
                break Ok(SolverSuccess(oc_counter, rms));
            }

            if rms == std::f64::NAN || rms == std::f64::INFINITY {
                break Err(SolverError::ConvergenceError(oc_counter));
            }

            oc_counter += 1;

            if oc_counter == self.max_iter {
                break Err(SolverError::MaxIterationError(oc_counter));
            }
        };

        result
    }
}
