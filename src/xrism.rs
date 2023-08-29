use crate::transforms::fourier_bessel_transform;
use ndarray::{Array, Array1, Array2, Array3, Axis, Zip};
use ndarray_linalg::Inverse;
use rustdct::DctPlanner;
use std::f64::consts::PI;

pub fn xrism_vv_equation(
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
) -> (Array3<f64>, Array3<f64>) {
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

    // Perform integral equation calculation in k-space
    // H = (I - W * C * P)^-1 * (W * C * W)
    Zip::from(hk.outer_iter_mut())
        .and(wk.outer_iter())
        .and(ck.outer_iter())
        .par_for_each(|mut hk_matrix, wk_matrix, ck_matrix| {
            let inverted_wcp = (&identity - wk_matrix.dot(&ck_matrix.dot(&p)))
                .inv()
                .expect("could not invert matrix");
            let wcw = wk_matrix.dot(&ck_matrix.dot(&wk_matrix));
            hk_matrix.assign(&inverted_wcp.dot(&wcw));
        });

    // Compute t(k) = h(k) - c(k)
    let tk = &hk - ck;

    // Transform t(k) -> t(r)
    Zip::from(tk.lanes(Axis(0)))
        .and(tr.lanes_mut(Axis(0)))
        .par_for_each(|tk_lane, mut tr_lane| {
            tr_lane.assign(&fourier_bessel_transform(
                ktor,
                &k,
                &r,
                &tk_lane.to_owned(),
                &plan,
            ));
        });

    // removing long-range component
    tr = tr - B * ur_lr;

    // return k-space total correlation and r-space indirect correlation functions
    (hk, tr)
}
