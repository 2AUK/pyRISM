use crate::closure::hyper_netted_chain;
use crate::data::DataRs;
use crate::xrism::xrism_vv;
use fftw::plan::*;
use fftw::types::*;
use ndarray_linalg::Solve;
use numpy::ndarray::{Array, Array1, Array2, Array3};
use log::info;

#[derive(Clone, Debug)]
pub struct MDIIS {
    // input parameters for solver
    pub m: usize,
    pub mdiis_damping: f64,
    pub picard_damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,

    // store original shape of array
    pub npts: usize,
    pub ns1: usize,
    pub ns2: usize,

    // arrays for MDIIS methods - to only be used in Rust code
    fr: Vec<Array1<f64>>,
    res: Vec<Array1<f64>>,
    rms_res: Vec<f64>,
}

impl MDIIS {
    fn step_picard(&mut self, curr: &Array3<f64>, prev: &Array3<f64>) -> Array3<f64> {
        // calculate difference between current and previous solutions from RISM equation
        let diff = curr.clone() - prev.clone();

        // push current flattened solution into MDIIS array
        self.fr.push(Array::from_iter(curr.clone().into_iter()));

        // push flattened difference into residual array
        self.res.push(Array::from_iter(diff.clone().into_iter()));

        // return Picard iteration step
        prev + self.picard_damping * diff
    }

    fn step_mdiis(
        &mut self,
        curr: &Array3<f64>,
        prev: &Array3<f64>,
        gr: &Array3<f64>,
    ) -> Array3<f64> {
        let mut a = Array2::zeros((self.m + 1, self.m + 1));
        let mut b = Array1::zeros(self.m + 1);

        let gr = Array::from_iter(gr.clone().into_iter());

        a[[self.m, self.m]] = 0.0;
        b[[self.m]] = -1.0;

        for i in 0..self.m + 1 {
            a[[i, self.m]] = -1.0;
            a[[self.m, i]] = -1.0;
        }

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
        self.fr.push(Array::from_iter(curr.clone().into_iter()));

        // push flattened difference into residual array
        self.res.push(Array::from_iter(diff.clone().into_iter()));

        self.fr.pop();
        self.res.pop();

        (c_a + self.mdiis_damping * min_res)
            .into_shape((self.npts, self.ns1, self.ns2))
            .expect("could not reshape array into original shape")
    }
}

impl MDIIS {
    pub fn new(
        m: usize,
        mdiis_damping: f64,
        picard_damping: f64,
        max_iter: usize,
        tolerance: f64,
        npts: usize,
        ns1: usize,
        ns2: usize,
    ) -> Self {
        MDIIS {
            m,
            mdiis_damping,
            picard_damping,
            max_iter,
            tolerance,
            npts,
            ns1,
            ns2,
            fr: Vec::new(),
            res: Vec::new(),
            rms_res: Vec::new(),
        }
    }

    pub fn solve(&mut self, data: &mut DataRs) {
        env_logger::init();
        info! {"Solving solvent-solvent RISM equation"};
        self.fr.clear();
        self.res.clear();
        self.rms_res.clear();
        let mut r2r: R2RPlan64 = R2RPlan::aligned(
            &[data.grid.npts],
            R2RKind::FFTW_RODFT11,
            Flag::ESTIMATE,
        )
        .expect("could not execute FFTW plan");
        let mut i = 0;
        while i < self.max_iter {
            let c_prev = data.cr.clone();
            xrism_vv(data, &mut r2r);
            let c_a = hyper_netted_chain(&data);
            let mut c_next;

            if self.fr.len() < self.m {
                c_next = self.step_picard(&c_a, &c_prev);
                let rmse = compute_rmse(
                    data.ns1,
                    data.ns2,
                    data.grid.npts,
                    &c_next,
                    &c_prev,
                );
                self.rms_res.push(rmse)
            } else {
                let gr = &data.tr + &c_a;
                c_next = self.step_mdiis(&c_a, &c_prev, &gr);
                let rmse = compute_rmse(
                    data.ns1,
                    data.ns2,
                    data.grid.npts,
                    &c_next,
                    &c_prev,
                );
                let rmse_min = self.rms_res.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let min_index = self
                    .rms_res
                    .iter()
                    .position(|x| *x == rmse_min)
                    .expect("could not find index of minimum in rms_res");
                if rmse > 10.0 * rmse_min {
                    c_next = self.fr[min_index]
                        .clone()
                        .into_shape((self.npts, self.ns1, self.ns2))
                        .expect("could not reshape array into original shape");
                    self.fr.clear();
                    self.res.clear();
                    self.rms_res.clear();
                }
                self.rms_res.push(rmse);
                self.rms_res.pop();
            }
            //println!("{:E}", c_next);
            data.cr = c_next.clone();
            let rmse = conv_rmse(
                data.ns1,
                data.ns2,
                data.grid.npts,
                data.grid.dr,
                &c_next,
                &c_prev,
            );
            if i % 10 == 0 {
                info!("Iteration: {}\nRMSE: {:E}", i, rmse);
            }

            if rmse < self.tolerance {
                info!("Converged at:\n\tIteration: {}\n\tRMSE: {:E}", i, rmse);
                break;
            }

            i += 1;

            if i == self.max_iter {
                info!(
                    "Max iteration reached at:\n\tIteration: {}\n\tRMSE: {:E}",
                    i, rmse
                );
                break;
            }
        }
    }
}

fn compute_rmse(
    ns1: usize,
    ns2: usize,
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
