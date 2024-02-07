use ndarray::{Array, Array1, Array3, Axis, Zip};
use std::f64::consts::PI;

use super::transforms::fourier_bessel_transform_fftw;

#[derive(Clone, Debug)]
pub struct Grid {
    pub npts: usize,
    pub radius: f64,
    pub dr: f64,
    pub dk: f64,
    pub rgrid: Array1<f64>,
    pub kgrid: Array1<f64>,
}

impl Grid {
    pub fn new(npts: usize, radius: f64) -> Self {
        let dr = radius / npts as f64;
        let dk = 2.0 * PI / (2.0 * npts as f64 * dr);
        Grid {
            npts,
            radius,
            dr,
            dk,
            rgrid: Array1::range(0.5, npts as f64, 1.0) * dr,
            kgrid: Array1::range(0.5, npts as f64, 1.0) * dk,
        }
    }

    pub fn nd_fbt_r2k(&self, in_arr: &Array3<f64>) -> Array3<f64> {
        let mut out_arr = Array::zeros(in_arr.raw_dim());
        Zip::from(in_arr.lanes(Axis(0)))
            .and(out_arr.lanes_mut(Axis(0)))
            .par_for_each(|in_lane, mut out_lane| {
                out_lane.assign(&fourier_bessel_transform_fftw(
                    2.0 * PI * self.dr,
                    &self.rgrid.view(),
                    &self.kgrid.view(),
                    &in_lane.to_owned(),
                ));
            });
        out_arr
    }

    pub fn nd_fbt_k2r(&self, in_arr: &Array3<f64>) -> Array3<f64> {
        let mut out_arr = Array::zeros(in_arr.raw_dim());
        Zip::from(in_arr.lanes(Axis(0)))
            .and(out_arr.lanes_mut(Axis(0)))
            .par_for_each(|in_lane, mut out_lane| {
                out_lane.assign(&fourier_bessel_transform_fftw(
                    self.dk / (4.0 * PI * PI),
                    &self.kgrid.view(),
                    &self.rgrid.view(),
                    &in_lane.to_owned(),
                ));
            });
        out_arr
    }
}
