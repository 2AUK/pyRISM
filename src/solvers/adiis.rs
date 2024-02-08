use crate::data::{configuration::solver::*, core::DataRs};
use crate::grids::transforms::fourier_bessel_transform_fftw;
use crate::iet::operator::Operator;
use crate::solvers::solver::Solver;
use log::{debug, info, trace};
use ndarray::{s, Array, Array1, Array2, Array3, Slice};
use ndarray::{Axis, Zip};
use ndarray_linalg::Solve;
use std::collections::VecDeque;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct ADIIS {
    // input parameters for solver
    pub m: usize,
    pub mdiis_damping: f64,
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,

    // arrays for MDIIS methods - to only be used in Rust code
    fr: VecDeque<Array1<f64>>,
    res: VecDeque<Array1<f64>>,
    rms_min: (f64, usize),
    a: Array2<f64>,
    curr_depth: usize,
    counter: usize,
    initial_step: bool,
}

impl ADIIS {
    pub fn new(settings: &SolverSettings) -> Self {
        let mdiis_settings = settings
            .clone()
            .mdiis_settings
            .expect("MDIIS settings not found");
        ADIIS {
            m: mdiis_settings.depth,
            mdiis_damping: mdiis_settings.damping,
            picard_damping: settings.picard_damping,
            max_iter: settings.max_iter,
            tolerance: settings.tolerance,
            fr: VecDeque::new(),
            res: VecDeque::new(),
            rms_min: (0.0, 0),
            a: Array::zeros((mdiis_settings.depth + 1, mdiis_settings.depth + 1)),
            curr_depth: 0,
            counter: 0,
            initial_step: true,
        }
    }

    fn step_adiis(&mut self, tr: &Array3<f64>) -> (f64, Array3<f64>) {
        // Check if we're in the initial step
        if self.initial_step {
            // Return an initial RMS of 0
            let rms = 1e20;

            self.fr.clear();
            self.res.clear();

            self.counter = 0;
            self.curr_depth = 0;

            self.rms_min = (0.0, 0);

            // Flatten input t(r) array
            let tr_flat = Array::from_iter(tr.clone());

            // Store initial solution for t(r)
            self.fr.push_front(tr_flat);

            // Initialise A and b arrays

            // Set the the first row and column to -1.0 and the the very first element to 0.0
            // This is so we can grow the A matrix and use the filled parts only.
            // The array is already zeroed so we only need to set the column/row to -1.0 bar the
            // first element

            self.a = Array::zeros((self.m + 1, self.m + 1));
            for i in 1..self.m + 1 {
                self.a[[0, i]] = -1.0;
                self.a[[i, 0]] = -1.0;
            }

            // No longer the initial step
            self.initial_step = false;

            // Return the t(k) - direct iteration
            return (rms, tr.to_owned());
        // If we're not in the intial step then start the MDIIS process
        } else {
            // Store the original 3D array shape
            let original_shape = tr.raw_dim();

            // Store number of points
            let n = (tr.dim().0 * tr.dim().1 * tr.dim().2) as f64;

            // Flatten input array
            let tr = Array::from_iter(tr.clone());

            // Compute squared sum
            let mut rmsnew = (&tr - self.fr.front().unwrap()).mapv(|x| x.powf(2.0)).sum();

            // Compute RMS
            rmsnew = (rmsnew / n).sqrt();

            if self.curr_depth == 0 {
                self.rms_min = (rmsnew, self.counter);
            }

            // println!(
            //     "RMS_new: {}, RMS_min: {}, {}",
            //     rmsnew,
            //     self.rms_min.0,
            //     rmsnew < self.rms_min.0
            // );
            // Check if the new RMS is less than the current minimum RMS
            if rmsnew < self.rms_min.0 {
                self.rms_min = (rmsnew, self.counter);
            }

            // println!(
            //     "RMS_new: {}, 10 * RMS_min: {}, {}",
            //     rmsnew,
            //     10.0 * self.rms_min.0,
            //     rmsnew > self.rms_min.0 * 10.0
            // );
            // Check if the new RMS is greather than the current minimum by a factor of 10
            if rmsnew > self.rms_min.0 * 10.0 && self.curr_depth > 0 {
                info!("MDIIS Restarting");
                self.curr_depth = 0;
                self.counter = 0;
                let min_fr = self.fr[self.rms_min.1].clone();
                let min_res = self.res[self.rms_min.1].clone();
                let rmsmin = min_res.dot(&min_res);
                self.fr.clear();
                self.res.clear();
                self.fr.push_front(min_fr.clone());
                self.res.push_front(min_res);

                return (
                    rmsmin,
                    min_fr
                        .into_shape(original_shape)
                        .expect("flattened array reshaped into original 3 dimensions"),
                );
            }

            let mut b = Array::zeros(self.m + 1);

            b[[0]] = -1.0;

            // Increment current depth counter
            self.curr_depth = std::cmp::min(self.curr_depth + 1, self.m);

            // Compute new residual
            let res = &tr - self.fr.front().unwrap();

            // Store residual
            self.res.push_front(res);

            if self.res.len() == self.m + 1 {
                self.res.pop_back();
            }

            // Construct new residual overlap matrix
            for i in 0..self.curr_depth {
                for j in 0..self.curr_depth {
                    self.a[[i + 1, j + 1]] = self.res[i].dot(&self.res[j]);
                }
            }

            let a_slice = self
                .a
                .slice(s![0..self.curr_depth + 1, 0..self.curr_depth + 1]);

            let b_slice = b.slice(s![0..self.curr_depth + 1]);

            let coefficients = a_slice
                .clone()
                .to_owned()
                .solve_into(b_slice.clone().to_owned())
                .expect("could not perform linear solve");

            let mut out = Array::zeros(tr.raw_dim());
            for i in 0..self.curr_depth {
                let fr_term = &self.fr[i] * coefficients[[i + 1]];
                let res_term = &self.res[i] * coefficients[[i + 1]];
                out = out + (fr_term + self.mdiis_damping * res_term);
            }

            // Store new input
            self.fr.push_front(out.clone());

            if self.fr.len() == self.m + 1 {
                self.fr.pop_back();
            }

            // Increment counter
            self.counter = (self.counter + 1) % self.m;

            return (
                rmsnew,
                out.to_owned()
                    .into_shape(original_shape)
                    .expect("flattened array reshaped into original 3 dimensions"),
            );
        }
    }
}

impl Solver for ADIIS {
    fn solve(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        // Iteration counter
        let mut i = 0;

        // We want to cycle this problem:
        // t(r) -> c(r) -> c(k) -> t(k) -> t'(r)
        // Current implementation focuses on a c(r) -> c'(r) cyclse
        // We can swap this by doing an initial RISM equation step to go from:
        // c(r) -> c(k) -> t(k) -> t(r)
        // And using this resulting t(r) as our initial guess.

        // Cycling c(r) -> t(r)
        (operator.eq)(problem);

        let result = loop {
            // Use closure to compute c'(r) from t(r)
            problem.correlations.cr = (operator.closure)(problem);

            // Use RISM equation to compute t'(r)
            (operator.eq)(problem);

            // Use MDIIS step to get new t(r)
            let (rms, t_new) = self.step_adiis(&problem.correlations.tr);

            problem.correlations.tr = t_new;

            trace!("Iteration: {} Convergence RMSE: {:.6E}", i, rms);

            if rms <= self.tolerance {
                self.initial_step = true;
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
