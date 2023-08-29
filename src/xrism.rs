use crate::transforms::fourier_bessel_transform;
use ndarray::{Array, Array1, Array2, Array3, Axis, Zip};
use rustdct::DctPlanner;
use std::f64::consts::PI;

fn xrism_vv_equation(
    ns: usize,
    npts: usize,
    r: Array1<f64>,
    k: Array1<f64>,
    dr: f64,
    dk: f64,
    cr: Array3<f64>,
    wk: Array3<f64>,
    p: Array1<f64>,
    B: f64,
    uk_lr: Array3<f64>,
    ur_lr: Array3<f64>,
) -> Array3<f64> {
    // Setting up prefactors for Fourier-Bessel transforms
    let rtok = 2.0 * PI * dr;
    let ktor = dk / (4.0 * PI * PI);

    // Starting FFT Plan
    let plan = DctPlanner::new().plan_dst4(npts);

    // Setting up arrays used in calculating XRISM equation
    let identity: Array2<f64> = Array::eye(ns);
    let mut ck = Array::zeros(cr.raw_dim());
    let mut hk = Array::zeros(cr.raw_dim());
    let mut tr = Array::zeros(cr.raw_dim());

    // Transforming c(r) -> c(k)
    Zip::from(cr.lanes(Axis(0)))
        .and(ck.lanes_mut(Axis(0)))
        .par_for_each(|cr_lane, mut ck_lane| {
            ck_lane.assign(&fourier_bessel_transform(
                rtok,
                &r,
                &k,
                &cr_lane.to_owned(),
                &plan,
            ));
        });
    
    // Adding long-range component back in
    ck = ck - B * uk_lr;

    ck
}
