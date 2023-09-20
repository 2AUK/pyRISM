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
    let arr = func * grid1;
    let mut input = arr.as_standard_layout();
    let mut output = Array1::zeros(input.raw_dim());
    r2r.r2r(input.as_slice_mut().unwrap() , output.as_slice_mut().unwrap())
        .expect("could not perform DST-IV operation");
    prefac * output / grid2
}
