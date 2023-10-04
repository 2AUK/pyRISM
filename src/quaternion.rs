use ndarray::{Array, Array1};

macro_rules! op_quat_quat {
    ($($path:ident)::+, $fn:ident) => {
        impl $($path)::+<Quaternion> for Quaternion {
            type Output = Quaternion;

            fn $fn(self, other: Quaternion) -> Self::Output {
                Quaternion([self.0[0].$fn(other.0[0]),
                    self.0[1].$fn(other.0[1]),
                    self.0[2].$fn(other.0[2]),
                    self.0[3].$fn(other.0[3])])
            }
        }
        impl $($path)::+<&Quaternion> for &Quaternion {
            type Output = Quaternion;

            fn $fn(self, other: &Quaternion) -> Self::Output {
Quaternion([self.0[0].$fn(other.0[0]),
                            self.0[1].$fn(other.0[1]),
    self.0[2].$fn(other.0[2]),
                            self.0[3].$fn(other.0[3])])
            }
        }
         impl $($path)::+<Quaternion> for &Quaternion {
            type Output = Quaternion;

            fn $fn(self, other: Quaternion) -> Self::Output {
                Quaternion([self.0[0].$fn(other.0[0]),
                            self.0[1].$fn(other.0[1]),
                            self.0[2].$fn(other.0[2]),
                            self.0[3].$fn(other.0[3])])
            }
        }
          impl $($path)::+<&Quaternion> for Quaternion {
            type Output = Quaternion;

            fn $fn(self, other: &Quaternion) -> Self::Output {
                Quaternion([self.0[0].$fn(other.0[0]),
                            self.0[1].$fn(other.0[1]),
                            self.0[2].$fn(other.0[2]),
                            self.0[3].$fn(other.0[3])])
            }
        }
    }
}

macro_rules! opassign_quat_quat {
    ($($path:ident)::+, $fn:ident) => {
        impl $($path)::+<Quaternion> for Quaternion {
            fn $fn(&mut self, other: Quaternion) {
                self.0[0].$fn(other.0[0]);
                self.0[1].$fn(other.0[1]);
                self.0[2].$fn(other.0[2]);
                self.0[3].$fn(other.0[3]);
            }
        }

        impl $($path)::+<&Quaternion> for Quaternion {
            fn $fn(&mut self,  other: &Quaternion) {
                self.0[0].$fn(other.0[0]);
                self.0[1].$fn(other.0[1]);
                self.0[2].$fn(other.0[2]);
                self.0[3].$fn(other.0[3]);
            }
        }
    }
}

op_quat_quat!(std::ops::Add, add);
op_quat_quat!(std::ops::Sub, sub);
opassign_quat_quat!(std::ops::AddAssign, add_assign);
opassign_quat_quat!(std::ops::SubAssign, sub_assign);

// Scalar first quaternion (w, x, y, z)
#[derive(Clone, Debug)]
pub struct Quaternion([f64; 4]);

impl Quaternion {
    pub fn new(input_arr: [f64; 4]) -> Self {
        Quaternion(input_arr)
    }

    pub fn rotate(&self, vec: &Array1<f64>) -> Array1<f64> {
        let w = self.0[0];
        let u_raw = self.0[1..4].to_vec();
        let vec = vec.to_owned();
        let u: Array1<f64> = Array::from_iter(u_raw.into_iter());
        let term1: Array1<f64> = 2.0 * u.dot(&vec) * &u;
        let term2: Array1<f64> = w * w - u.dot(&u) * &vec;
        let term3: Array1<f64> = 2.0 * w * cross_product(&u, &vec);

        term1 + term2 + term3
    }

    pub fn from_axis_angle(angle: f64, axis: &Array1<f64>) -> Self {
        let magnitude = axis.mapv(|x| x.powf(2.0)).sum().sqrt();
        let scalar_w = (angle / 2.0).cos();
        match magnitude < 1e-9 {
            true => Quaternion([scalar_w, 0.0, 0.0, 0.0]),
            _ => {
                let quat1to3 = axis / magnitude * (angle / 2.0).sin();
                Quaternion([scalar_w, quat1to3[0], quat1to3[1], quat1to3[2]])
            }
        }
    }
}

pub fn cross_product(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut out_arr = Array::zeros(3);
    out_arr[[0]] = a[[1]] * b[[2]] - a[[2]] * b[[1]];
    out_arr[[1]] = a[[2]] * b[[0]] - a[[0]] * b[[2]];
    out_arr[[2]] = a[[0]] * b[[1]] - a[[1]] * b[[0]];
    out_arr
}

impl std::ops::Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    fn mul(self, other: Quaternion) -> Self::Output {
        let mut out_arr: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
        let a = self.0;
        let b = other.0;
        out_arr[0] = a[0] * b[0] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        out_arr[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[3];
        out_arr[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
        out_arr[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];

        Quaternion(out_arr)
    }
}

impl std::ops::Mul<&Quaternion> for Quaternion {
    type Output = Quaternion;

    fn mul(self, other: &Quaternion) -> Self::Output {
        let mut out_arr: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
        let a = self.0;
        let b = other.0;
        out_arr[0] = a[0] * b[0] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        out_arr[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[3];
        out_arr[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
        out_arr[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];

        Quaternion(out_arr)
    }
}

impl std::ops::Mul<&Quaternion> for &Quaternion {
    type Output = Quaternion;

    fn mul(self, other: &Quaternion) -> Self::Output {
        let mut out_arr: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
        let a = self.0;
        let b = other.0;
        out_arr[0] = a[0] * b[0] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        out_arr[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[3];
        out_arr[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
        out_arr[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];

        Quaternion(out_arr)
    }
}

impl std::ops::Mul<Quaternion> for &Quaternion {
    type Output = Quaternion;

    fn mul(self, other: Quaternion) -> Self::Output {
        let mut out_arr: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
        let a = self.0;
        let b = other.0;
        out_arr[0] = a[0] * b[0] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        out_arr[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[3];
        out_arr[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
        out_arr[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];

        Quaternion(out_arr)
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_quat() {
        let quat = Quaternion::new([1.0, 2.0, 3.0, 4.0]);
        println!("Print: {:?}", quat);
    }

    #[test]
    fn test_rotation() {
        // Quaternion representing reflection in y & z axes
        let quat = Quaternion::new([0.0, 1.0, 0.0, 0.0]);
        // Rotation matrix performing same operation
        let rot_mat = Array::from_diag(&arr1(&[1.0, -1.0, -1.0]));
        // Input vec
        let vec = Array::from_vec(vec![1.0, 1.0, 1.0]);

        // Expect output -> [1.0, -1.0, -1.0]
        println!("Rotation Matrix: {:?}", rot_mat.dot(&vec));
        println!("Quaternion: {:?}", quat.rotate(&vec));
    }
}
//
// #[test]
// fn test_quat_addition() {
//     let quat1 = Quaternion::new([1.0, 2.0, 3.0, 4.0]);
//     let quat2 = Quaternion::new([4.0, 3.0, 2.0, 1.0]);
//
//     let quat_result = quat1 + &quat2;
//
//     println!("Add: {:?}", quat_result);
// }
//
// #[test]
// fn test_quat_subtraction() {
//     let quat1 = Quaternion::new([5.0, 5.0, 5.0, 5.0]);
//     let quat2 = Quaternion::new([1.0, 2.0, 3.0, 4.0]);
//
//     let quat_result = &quat1 - quat2;
//
//     println!("Sub: {:?}", quat_result);
// }

// #[test]
// fn test_quat_scalar_multiplication() {
//     let quat1 = Quaternion::new([1.0, 2.0, 3.0, 4.0]);
//
//     // let quat_result_lhs = &quat1 * 2.0;
//     // let quat_result_rhs = 3.0 * &quat1;
// }
// }
