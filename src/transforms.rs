use fftw::array::AlignedVec;
use fftw::plan::*;
use ndarray::{Array1, ArrayView1};
use rustdct::TransformType4;
use std::sync::Arc;

type FFTPlan = Arc<dyn TransformType4<f64>>;

pub fn fourier_bessel_transform(
    prefac: f64,
    grid1: &ArrayView1<f64>,
    grid2: &ArrayView1<f64>,
    func: &Array1<f64>,
    plan: &FFTPlan,
) -> Array1<f64> {
    let mut buffer = (func * grid1).to_vec();
    plan.process_dst4(&mut buffer);
    let fac = grid2.mapv(|v| prefac / v);
    fac * Array1::from_vec(buffer)
}

pub fn fourier_bessel_transform_fftw(
    prefac: f64,
    grid1: &ArrayView1<f64>,
    grid2: &ArrayView1<f64>,
    func: &Array1<f64>,
    r2r: &mut R2RPlan64,
) -> Array1<f64> {
    let n = grid1.len();

    let mut input = AlignedVec::new(n);
    for i in 0..n {
        input[i] = (func * grid1)[[i]];
    }

    let mut output = AlignedVec::new(n);

    let mut out_arr = Array1::zeros(func.raw_dim());
    r2r.r2r(&mut input, &mut output)
        .expect("could not perform DST-IV operation");
    for i in 0..n {
        out_arr[[i]] = prefac * output[i] / grid2[i];
    }
    out_arr
}
