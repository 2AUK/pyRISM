use fftw::plan::*;
use fftw::types::*;
use ndarray::{Array1, ArrayView1};

pub fn fourier_bessel_transform_fftw(
    prefac: f64,
    grid1: &ArrayView1<f64>,
    grid2: &ArrayView1<f64>,
    func: &Array1<f64>,
) -> Array1<f64> {
    let arr = grid1 * func;
    let mut r2r: R2RPlan64 =
        R2RPlan::aligned(&[grid1.len()], R2RKind::FFTW_RODFT11, Flag::ESTIMATE)
            .expect("could not execute FFTW plan");
    let mut input = arr.as_standard_layout();
    let mut output = Array1::zeros(input.raw_dim());
    r2r.r2r(
        input.as_slice_mut().unwrap(),
        output.as_slice_mut().unwrap(),
    )
    .expect("could not perform DST-IV operation");
    prefac * output / grid2
}
