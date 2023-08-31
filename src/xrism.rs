use crate::transforms::fourier_bessel_transform_fftw;
use ndarray::{Array, Array1, Array2, Array3, Axis, Zip, ArrayView1, ArrayView3, ArrayView2};
use ndarray_linalg::Inverse;
use std::f64::consts::PI;

pub fn xrism_vv_equation(
    ns: usize,
    npts: usize,
    r: ArrayView1<f64>,
    k: ArrayView1<f64>,
    dr: f64,
    dk: f64,
    cr: ArrayView3<f64>,
    wk: ArrayView3<f64>,
    p: ArrayView2<f64>,
    B: f64,
    uk_lr: ArrayView3<f64>,
    ur_lr: ArrayView3<f64>,
) -> (Array3<f64>, Array3<f64>) {
    // Setting up prefactors for Fourier-Bessel transforms
    let rtok = 2.0 * PI * dr;
    let ktor = dk / (4.0 * PI * PI);

    // Starting FFT Plan
    // let plan = DctPlanner::new().plan_dst4(npts);

    // Setting up arrays used in calculating XRISM equation
    let identity: Array2<f64> = Array::eye(ns);
    let mut ck = Array::zeros(cr.raw_dim());
    let mut hk = Array::zeros(cr.raw_dim());
    let mut tr = Array::zeros(cr.raw_dim());

    // Transforming c(r) -> c(k)
    Zip::from(cr.lanes(Axis(0)))
        .and(ck.lanes_mut(Axis(0)))
        .for_each(|cr_lane, mut ck_lane| {
            ck_lane.assign(&fourier_bessel_transform_fftw(
                rtok,
                &r,
                &k,
                &cr_lane.to_owned(),
            ));
        });

    // Adding long-range component back in
    ck = ck - B * uk_lr.to_owned();

    // Perform integral equation calculation in k-space
    // H = (I - W * C * P)^-1 * (W * C * W)
    Zip::from(hk.outer_iter_mut())
        .and(wk.outer_iter())
        .and(ck.outer_iter())
        .for_each(|mut hk_matrix, wk_matrix, ck_matrix| {
            let iwcp = &identity - wk_matrix.dot(&ck_matrix.dot(&p));
            let inverted_iwcp = (iwcp)
                .inv()
                .expect("could not invert matrix: {iwcp}");
            let wcw = wk_matrix.dot(&ck_matrix.dot(&wk_matrix));
            hk_matrix.assign(&inverted_iwcp.dot(&wcw));
        });

    // Compute t(k) = h(k) - c(k)
    let tk = &hk - ck;

    // Transform t(k) -> t(r)
    Zip::from(tk.lanes(Axis(0)))
        .and(tr.lanes_mut(Axis(0)))
        .for_each(|tk_lane, mut tr_lane| {
            tr_lane.assign(&fourier_bessel_transform_fftw(
                ktor,
                &k,
                &r,
                &tk_lane.to_owned(),
            ));
        });

    // removing long-range component
    tr = tr - B * ur_lr.to_owned();

    // return k-space total correlation and r-space indirect correlation functions
    (hk, tr)
}
