use crate::data::configuration::solver::*;
use crate::data::DataRs;
use crate::grids::transforms::fourier_bessel_transform_fftw;
use crate::iet::operator::Operator;
use crate::solvers::solver::Solver;
use log::{info, trace};
use ndarray::{Axis, Zip};
use ndarray_linalg::Solve;
use numpy::ndarray::{Array, Array1, Array2, Array3};
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
    rms_res: VecDeque<f64>,
    curr_depth: usize,
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
            rms_res: VecDeque::new(),
            curr_depth: 0,
        }
    }

    fn step_mdiis(&mut self, curr: &Array3<f64>, res: &Array3<f64>) -> Array1<f64> {
        let mut a = Array2::zeros((self.curr_depth + 1, self.curr_depth + 1));
        let mut b = Array1::zeros(self.curr_depth + 1);
        self.fr.push_back(Array::from_iter(curr.clone()));

        // push flattened difference into residual array
        self.res.push_back(Array::from_iter(res.clone()));

        b[[self.curr_depth]] = -1.0;

        for i in 0..self.curr_depth + 1 {
            a[[i, self.curr_depth]] = -1.0;
            a[[self.curr_depth, i]] = -1.0;
        }

        a[[self.curr_depth, self.curr_depth]] = 0.0;

        for i in 0..self.curr_depth {
            for j in 0..self.curr_depth {
                a[[i, j]] = self.res[i].dot(&self.res[j]);
            }
        }

        let coefficients = a.solve_into(b).expect("could not perform linear solve");
        //println!("coefs: {:?}", coefficients);

        let mut c_a: Array1<f64> = Array::zeros(self.fr[0].raw_dim());
        let mut min_res: Array1<f64> = Array::zeros(self.fr[0].raw_dim());
        for i in 0..self.curr_depth {
            let modified_fr = &self.fr[i] * coefficients[i];
            let modified_res = &self.res[i] * coefficients[i];
            c_a += &modified_fr;
            min_res += &modified_res;
        }
        if self.curr_depth == self.m {
            self.fr.pop_front();
            self.res.pop_front();
        }
        c_a + self.mdiis_damping * min_res
    }
}

impl Solver for ADIIS {
    fn solve(
        &mut self,
        problem: &mut DataRs,
        operator: &Operator,
    ) -> Result<SolverSuccess, SolverError> {
        info! {"Solving RISM equation"};
        self.fr.clear();
        self.res.clear();
        self.rms_res.clear();
        let shape = problem.correlations.cr.dim();
        let mut i = 0;
        let dr = problem.grid.dr;
        let rtok = 2.0 * PI * dr;
        let r = problem.grid.rgrid.clone();
        let k = problem.grid.kgrid.clone();

        problem.correlations.cr = -problem.system.beta * problem.interactions.ur_lr.clone();
        Zip::from(problem.correlations.cr.lanes_mut(Axis(0))).par_for_each(|mut elem| {
            elem.assign(&(&elem * &r.view()));
        });
        let result = loop {
            let _c_prev = problem.correlations.cr.clone();
            (operator.eq)(problem);
            let c_a = (operator.closure)(problem);
            let hr = {
                let mut out = Array::zeros(c_a.raw_dim());
                Zip::from(problem.correlations.hk.lanes(Axis(0)))
                    .and(out.lanes_mut(Axis(0)))
                    .par_for_each(|cr_lane, mut ck_lane| {
                        ck_lane.assign(&fourier_bessel_transform_fftw(
                            rtok,
                            &r.view(),
                            &k.view(),
                            &cr_lane.to_owned(),
                        ));
                    });
                out
            };
            let gr = &c_a + &problem.correlations.tr + 1.0;
            // let r = &r.broadcast((r.len(), 0, 0)).unwrap();
            // println!("{:?}", r.shape());
            let res = &gr - &hr - 1.0;
            // Zip::from(res.lanes_mut(Axis(0))).par_for_each(|mut elem| {
            //     elem.assign(&(&elem * &r.view()));
            // });
            let mut c_next;
            self.curr_depth = std::cmp::min(self.curr_depth + 1, self.m);
            c_next = self
                .step_mdiis(&c_a, &res)
                .into_shape(shape)
                .expect("could not reshape array into original shape");
            // Zip::from(c_next.lanes_mut(Axis(0))).par_for_each(|mut elem| {
            //     elem.assign(&(&elem / &r.view()));
            // });
            let rmse = conv_rmse(&res);
            //println!("\tMDIIS RMSE: {}", rmse);
            if self.curr_depth > 1 {
                let rmse_min = self.rms_res.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let min_index = self
                    .rms_res
                    .iter()
                    .position(|x| *x == rmse_min)
                    .expect("could not find index of minimum in rms_res");
                if rmse > 10.0 * rmse_min {
                    trace!("MDIIS restarting");
                    self.curr_depth = 0;
                    c_next = self.fr[min_index]
                        .clone()
                        .into_shape(shape)
                        .expect("could not reshape array into original shape");
                    self.fr.clear();
                    self.res.clear();
                    self.rms_res.clear();
                }
            }
            self.rms_res.push_back(rmse);
            if self.curr_depth == self.m {
                self.rms_res.pop_front();
            }
            problem.correlations.cr = c_next.clone();
            let rmse = conv_rmse(&res);

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

fn conv_rmse(res: &Array3<f64>) -> f64 {
    let denom = 1.0 / res.len() as f64;
    (res.mapv(|x| x.powf(2.0)).sum() * denom).sqrt()
}
