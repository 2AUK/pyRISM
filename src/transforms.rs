use ndarray::{ArrayView1, Array1};
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

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{Array, Axis};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn random_test() {
        let array = Array::random((16, 2, 2), Uniform::new(0., 10.));
        println!("{array}");
        array.lanes(Axis(0)).into_iter().for_each(|a| {
            println!("{a}");
            println!("{}", a.len());
        });
    }
}
