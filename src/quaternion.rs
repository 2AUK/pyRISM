use ndarray::{Array, Array1};

#[derive(Clone, Debug)]
pub struct Quaternion(Array1<f64>);

impl Quaternion {
    pub fn new(input_arr: [f64; 4]) -> Self {
        Quaternion(Array::from_iter(input_arr.into_iter()))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_quat() {
        let quat = Quaternion::new([1.0, 2.0, 3.0, 4.0]);
        println!("{:?}", quat);
    }
}
