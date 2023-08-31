use ndarray::{Array3, ArrayView3};

pub fn hyper_netted_chain(b: f64, u: ArrayView3<f64>, t: ArrayView3<f64>) -> Array3<f64> {
    (-b * u.to_owned() + t).mapv(|a| a.exp()) - 1.0 - t
}
