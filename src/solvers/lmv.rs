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
        ck0: &Array3<f64>,
        tk_delta: &Array3<f64>,
        k: &Array1<f64>,
        tk_curr: &Array3<f64>,
        wk: &Array3<f64>,
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
        let mut sum = 0.0;
        for v in 0..self.nbasis {
            for n in 0..self.nbasis {
                for a in 0..ns1 {
                    for b in 0..ns2 {
                        sum += cjk[[v, n, a, b]] * tk_delta[[n, a, b]];
                    }
                }
            }
        }

        let mut summed_term = Array::zeros((self.nbasis, ns1, ns2));
        for v in 0..self.nbasis {
            for a in 0..ns1 {
                for b in 0..ns2 {
                    let cjk_abj = cjk.slice(s![.., v, a, b]).sum();
                    summed_term[[v, a, b]] = cjk_abj * tk_delta[[v, a, b]];
                }
            }
        }
        let ck = ck0 + sum;

        for v in 0..self.nbasis {
            for a in 0..ns1 {
                for b in 0..ns2 {
                    // println!("{}", v * ns1 * ns2 + a * ns2 + b);
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

                                let kron_vn = {
                                    if v == n {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                };

                                let kron_cd = {
                                    if c == d {
                                        1.0277
                                    } else {
                                        0.0
                                    }
                                };

                                let kron_acbd = kron_ac * kron_bd;

                                // matrix
                                //     [[v * ns1 * ns2 + a * ns2 + b, n * ns1 * ns2 + c * ns2 + d]] =
                                //     k[[v]] * tk_delta[[n, a, b]]
                                //         - (tk_curr[[v, a, c]] + ck[[v, a, c]] + wk[[v, a, c]])
                                //             * (tk_curr[[n, d, b]] + ck[[n, d, b]] + wk[[n, d, b]])
                                //             * kron_ac
                                //             * kron_bd;
                                matrix
                                    [[v * ns1 * ns2 + a * ns2 + b, n * ns1 * ns2 + c * ns2 + d]] =
                                    k[[v]] * tk_delta[[v, a, b]]
                                        - ((tk_curr[[v, a, c]] + ck[[v, a, c]] + wk[[v, a, c]])
                                            * (tk_curr[[n, b, d]] * ck[[n, b, d]] * wk[[n, b, d]])
                                            - kron_acbd)
                                            * summed_term[[n, c, d]];
                                // matrix
                                //     [[v * ns1 * ns2 + a * ns2 + b, n * ns1 * ns2 + c * ns2 + d]] =
                                //     kron_acbd * (kron_vn + cjk[[v, n, a, b]])
                                //         - invwc1w[[v, a, c]]
                                //             * cjk[[v, n, c, d]]
                                //             * invwc1w[[n, b, d]];
                                // -k[[v]] * tk_delta[[v, a, b]];
                                // matrix
                                //     [[v * ns1 * ns2 + a * ns2 + b, n * ns1 * ns2 + c * ns2 + d]] =
                                //     (invwc1w[[v, a, c]] * invwc1w[[n, b, d]]
                                //         + invwc1w[[v, a, d]]
                                //             * invwc1w[[n, c, b]]
                                //             * (1.0 - kron_cd)
                                //         - kron_ac * kron_bd)
                                //         * ck[[n, c, d]]
                                //         - kron_ac * kron_bd * kron_vn;
                                // matrix
                                //     [[v * ns1 * ns2 + a * ns2 + b, n * ns1 * ns2 + c * ns2 + d]] = k
                                //     [[v]]
                                //     * (tk_delta[[v, a, b]]
                                //         - invwc1w[[v, a, c]]
                                //             * (2.0 + invwc1w[[n, b, d]])
                                //             * ck[[n, c, d]]);
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

    fn split_problem(&self, problem: &DataRs) -> (DataRs, DataRs) {
        let mut coarse_problem = problem.clone();
        let mut fine_problem = problem.clone();

        (fine_problem.correlations.tk, coarse_problem.correlations.tk) =
            self.split_function(&problem.correlations.tk);
        (fine_problem.correlations.tr, coarse_problem.correlations.tr) =
            self.split_function(&problem.correlations.tr);
        (fine_problem.correlations.hr, coarse_problem.correlations.hr) =
            self.split_function(&problem.correlations.hr);
        (fine_problem.correlations.hk, coarse_problem.correlations.hk) =
            self.split_function(&problem.correlations.hk);
        (fine_problem.correlations.cr, coarse_problem.correlations.cr) =
            self.split_function(&problem.correlations.cr);

        (fine_problem, coarse_problem)
    }

    fn split_function(&self, in_arr: &Array3<f64>) -> (Array3<f64>, Array3<f64>) {
        let (npts, ns1, ns2) = in_arr.dim();
        let (mut coarse_arr, mut fine_arr) = (
            Array::zeros(in_arr.raw_dim()),
            Array::zeros(in_arr.raw_dim()),
        );
        for l in 0..npts {
            for a in 0..ns1 {
                for b in 0..ns2 {
                    if l < self.nbasis {
                        fine_arr[[l, a, b]] = in_arr[[l, a, b]];
                    } else {
                        coarse_arr[[l, a, b]] = in_arr[[l, a, b]];
                    }
                }
            }
        }
        (fine_arr, coarse_arr)
    }

    fn recompose_problem(&self, problem: &mut DataRs, coarse: &DataRs, fine: &DataRs) {
        (problem.correlations.tk) =
            self.recompose_function(&coarse.correlations.tk, &fine.correlations.tk);
        (problem.correlations.tr) =
            self.recompose_function(&coarse.correlations.tr, &fine.correlations.tr);
        (problem.correlations.hk) =
            self.recompose_function(&coarse.correlations.hk, &fine.correlations.hk);
        (problem.correlations.cr) =
            self.recompose_function(&coarse.correlations.cr, &fine.correlations.cr);
    }

    fn recompose_function(&self, coarse_arr: &Array3<f64>, fine_arr: &Array3<f64>) -> Array3<f64> {
        let (npts, ns1, ns2) = coarse_arr.dim();
        let mut out_arr = Array::zeros(coarse_arr.raw_dim());
        for l in 0..npts {
            for a in 0..ns1 {
                for b in 0..ns2 {
                    if l < self.nbasis {
                        out_arr[[l, a, b]] = fine_arr[[l, a, b]];
                    } else {
                        out_arr[[l, a, b]] = coarse_arr[[l, a, b]];
                    }
                }
            }
        }
        out_arr
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

        // We also want to split the problem into a coarse and fine problem to iterature
        // separately.

        // Set up the coarse problem
        let coarse_problem = problem.clone();

        // Generate initial guess for t(r)
        (operator.eq)(problem);

        let result = loop {
            let rms = 1E20;
            // Store previous t(r)
            let tr_prev = problem.correlations.tr.clone();

            // Store previous t(k)
            let tk_prev = problem.correlations.tk.clone();

            // Compute c'(r) from closure
            problem.correlations.cr = (operator.closure)(problem);

            let ck = problem.grid.nd_fbt_k2r(&problem.correlations.cr);

            // Compute new t(r)
            (operator.eq)(problem);

            // Store current t(k)
            let tk_curr = problem.correlations.tk.clone();

            // Compute difference between the current and previous t(k)
            let mut delta_tk = &tk_curr - &tk_prev;

            // Compute dc'(r)/dt(r)
            let dcrtr = (operator.closure_der)(problem);

            // Compute coefficients from derivative
            let cjk = self.tabulate_coefficients(&dcrtr);

            let mut tk_ic_prev = tk_curr.clone();

            let mut ic_counter = 0;
            let mut tk_ic_curr = Array::zeros(tk_curr.raw_dim());
            // Start innner convergence loop
            let tk_ic_new = loop {
                // Compute Jacobian and vector
                let (vec, mat) = self.jacobian(
                    &cjk,
                    &ck,
                    &delta_tk,
                    &problem.grid.kgrid,
                    &tk_ic_curr,
                    &problem.data_a.borrow().wk.clone(),
                    &problem.correlations.invwc1wk,
                );

                println!("{:+.4e}", mat);

                let new_delta_tk_flat = mat
                    .solve_into(vec)
                    .expect("linear solve in NR step for new delta t(k)");

                let rms = new_delta_tk_flat.mapv(|x| x.powf(2.0)).sum().sqrt()
                    / (self.nbasis * ns1 * ns2) as f64;

                tk_ic_curr = Array::zeros(tk_curr.raw_dim());
                for v in 0..self.nbasis {
                    for a in 0..ns1 {
                        for b in 0..ns2 {
                            tk_ic_curr[[v, a, b]] = tk_ic_curr[[v, a, b]]
                                + self.picard_damping
                                    * (new_delta_tk_flat[[v * ns1 * ns2 + a * ns2 + b]]
                                        / problem.grid.kgrid[[v]]);
                        }
                    }
                }
                trace!("IC Iteration: {} Convergence RMSE: {:.6E}", ic_counter, rms);

                if rms < 1e-5 {
                    break tk_ic_curr;
                }

                ic_counter += 1;

                delta_tk = tk_ic_curr.clone() - tk_ic_prev.clone();
                tk_ic_prev = tk_ic_curr.clone();
            };

            let tr_curr = problem.grid.nd_fbt_k2r(&(tk_curr + tk_ic_new));

            let tr_new = 0.5 * tr_curr + (1.0 - 0.5) * &tr_prev;

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
