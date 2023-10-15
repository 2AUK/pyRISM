use crate::data::DataRs;
use crate::operator::Operator;
use crate::solver::{Solver, SolverError, SolverSettings, SolverSuccess};
use log::{info, trace, warn};
use ndarray_linalg::Solve;
use numpy::ndarray::{Array, Array1, Array2, Array3};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct MDIIS {
    // input parameters for solver
    pub m: usize,
    pub mdiis_damping: f64,
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,

    // arrays for MDIIS methods - to only be used in Rust code
    fr: VecDeque<Array1<f64>>,
    res: VecDeque<Array1<f64>>,
    rms_res: VecDeque<f64>,
}

impl MDIIS {
    pub fn new(settings: &SolverSettings) -> Self {
        let mdiis_settings = settings
            .clone()
            .mdiis_settings
            .expect("MDIIS settings not found");
        MDIIS {
            m: mdiis_settings.depth,
            mdiis_damping: mdiis_settings.damping,
            picard_damping: settings.picard_damping,
            max_iter: settings.max_iter,
            tolerance: settings.tolerance,
            fr: VecDeque::new(),
            res: VecDeque::new(),
            rms_res: VecDeque::new(),
        }
    }

    fn step_picard(&mut self, curr: &Array3<f64>, prev: &Array3<f64>) -> Array3<f64> {
        // calculate difference between current and previous solutions from RISM equation
        let diff = curr.clone() - prev.clone();

        // push current flattened solution into MDIIS array
        self.fr
            .push_back(Array::from_iter(curr.clone().into_iter()));

        // push flattened difference into residual array
        self.res
            .push_back(Array::from_iter(diff.clone().into_iter()));

        // return Picard iteration step
        prev + self.picard_damping * diff
    }

    fn step_mdiis(
        &mut self,
        curr: &Array3<f64>,
        prev: &Array3<f64>,
        gr: &Array3<f64>,
    ) -> Array1<f64> {
        let mut a = Array2::zeros((self.m + 1, self.m + 1));
        let mut b = Array1::zeros(self.m + 1);

        let gr = Array::from_iter(gr.clone().into_iter());

        b[[self.m]] = -1.0;

        for i in 0..self.m + 1 {
            a[[i, self.m]] = -1.0;
            a[[self.m, i]] = -1.0;
        }

        a[[self.m, self.m]] = 0.0;

        for i in 0..self.m {
            for j in 0..self.m {
                a[[i, j]] = self.res[i].dot(&self.res[j]);
            }
        }

        let coefficients = a.solve_into(b).expect("could not perform linear solve");

        let mut c_a: Array1<f64> = Array::zeros(self.fr[0].raw_dim());
        let mut min_res: Array1<f64> = Array::zeros(self.fr[0].raw_dim());
        let denom = (1.0 + gr.mapv(|a| a.powf(2.0))).mapv(f64::sqrt);
        for i in 0..self.m {
            let modified_fr = &self.fr[i] * coefficients[i];
            let modified_res = &self.res[i] * coefficients[i] / &denom;
            c_a += &modified_fr;
            min_res += &modified_res;
        }

        // calculate difference between current and previous solutions from RISM equation
        let diff = curr.clone() - prev.clone();

        // push current flattened solution into MDIIS array
        self.fr
            .push_back(Array::from_iter(curr.clone().into_iter()));

        // push flattened difference into residual array
        self.res
            .push_back(Array::from_iter(diff.clone().into_iter()));

        self.fr.pop_front();
        self.res.pop_front();

        c_a + self.mdiis_damping * min_res
    }
}

impl Solver for MDIIS {
    fn solve(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        info! {"Solving solvent-solvent RISM equation"};
        self.fr.clear();
        self.res.clear();
        self.rms_res.clear();
        let shape = problem.correlations.cr.dim();
        let (npts, ns1, ns2) = shape;
        let mut i = 0;

        let result = loop {
            //println!("Iteration: {}", i);
            let c_prev = problem.correlations.cr.clone();
            (operator.eq)(problem);
            let c_a = (operator.closure)(&problem);
            let mut c_next;

            if self.fr.len() < self.m {
                //println!("Picard Step");
                c_next = self.step_picard(&c_a, &c_prev);
                let rmse = compute_rmse(ns1, ns2, npts, &c_a, &c_prev);
                //println!("\tMDIIS RMSE: {}", rmse);
                self.rms_res.push_back(rmse)
            } else {
                let gr = &c_a + &problem.correlations.tr;
                //println!("MDIIS Step");
                c_next = self
                    .step_mdiis(&c_a, &c_prev, &gr)
                    .into_shape(shape)
                    .expect("could not reshape array into original shape");
                let rmse = compute_rmse(ns1, ns2, npts, &c_a, &c_prev);
                //println!("\tMDIIS RMSE: {}", rmse);
                let rmse_min = self.rms_res.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let min_index = self
                    .rms_res
                    .iter()
                    .position(|x| *x == rmse_min)
                    .expect("could not find index of minimum in rms_res");
                if rmse > 10.0 * rmse_min {
                    trace!("MDIIS restarting");
                    c_next = self.fr[min_index]
                        .clone()
                        .into_shape(shape)
                        .expect("could not reshape array into original shape");
                    self.fr.clear();
                    self.res.clear();
                    self.rms_res.clear();
                }
                self.rms_res.push_back(rmse);
                self.rms_res.pop_front();
            }
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
        };
        result
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
