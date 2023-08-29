use rustdct::TransformType4;
use std::sync::Arc;
use ndarray::Array1;

type FFTPlan = Arc<dyn TransformType4<f64>>;

pub fn dfbt(prefac: f64, grid1: Array1<f64>, grid2: Array1<f64>, func: Array1<f64>, plan: &FFTPlan) -> Array1<f64> {
    let mut buffer = (func * grid1).to_vec();
    plan.process_dst4(&mut buffer);
    prefac * Array1::from_vec(buffer) / grid2
}