use ndarray::{arr1, Array, Array1, Array2};

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

    pub fn conj(&self) -> Self {
        Quaternion::new([self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }

    pub fn rotate(&self, vec: &Array1<f64>) -> Array1<f64> {
        let _quat_raw_vec = [0.0, vec[0], vec[1], vec[2]];
        let q_vec = Quaternion::new(_quat_raw_vec);
        let q_temp = self * q_vec;
        let q_conj = self.conj();
        let q_vec = q_temp * q_conj;
        arr1(&[q_vec.0[1], q_vec.0[2], q_vec.0[3]])
    }

    pub fn from_axis_angle(angle: f64, axis: &Array1<f64>) -> Self {
        let magnitude = axis.mapv(|x| x.powf(2.0)).sum().sqrt();
        let scalar_w = (angle / 2.0).cos();
        match magnitude == 0.0 {
            true => Quaternion([scalar_w, 0.0, 0.0, 0.0]),
            _ => {
                let quat1to3 = axis / magnitude * (angle / 2.0).sin();
                Quaternion([scalar_w, quat1to3[0], quat1to3[1], quat1to3[2]])
            }
        }
    }

    pub fn to_rotation_matrix(&self) -> Array2<f64> {
        let mut out_arr = Array::zeros((3, 3));
        let (q0, q1, q2, q3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        out_arr[[0, 0]] = 2.0 * (q0 * q0 + q1 * q1) - 1.0;
        out_arr[[0, 1]] = 2.0 * (q1 * q2 - q0 * q3);
        out_arr[[0, 2]] = 2.0 * (q1 * q3 + q0 * q2);
        out_arr[[1, 0]] = 2.0 * (q1 * q2 + q0 * q3);
        out_arr[[1, 1]] = 2.0 * (q0 * q0 + q2 * q2) - 1.0;
        out_arr[[1, 2]] = 2.0 * (q2 * q3 - q0 * q1);
        out_arr[[2, 0]] = 2.0 * (q1 * q3 - q0 * q2);
        out_arr[[2, 1]] = 2.0 * (q2 * q3 + q0 * q1);
        out_arr[[2, 2]] = 2.0 * (q0 * q0 + q3 * q3) - 1.0;
        out_arr
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
        out_arr[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
        out_arr[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
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
        out_arr[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
        out_arr[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
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
        out_arr[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
        out_arr[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
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
        out_arr[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
        out_arr[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
        out_arr[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
        out_arr[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];

        Quaternion(out_arr)
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;
    use std::f64::consts::PI;

    const PRECISION: f64 = f64::EPSILON;

    #[test]
    fn test_to_rotation_matrix() {
        let quat = Quaternion::from_axis_angle(PI / 2.0, &arr1(&[0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_rotation() {
        // Quaternion representing 90 degree rotation in z-axis
        let quat = Quaternion::from_axis_angle(PI / 2.0, &arr1(&[0.0, 0.0, 1.0]));
        // Input vec
        let vec = Array::from_vec(vec![1.0, 0.5, 0.5]);

        let known_result = arr1(&[-0.5, 1.0, 0.5]);
        let calculated_result = quat.rotate(&vec);

        assert_relative_eq!(
            known_result,
            calculated_result,
            max_relative = PRECISION,
            epsilon = PRECISION
        );
    }

    #[test]
    fn test_from_angle_axis() {
        let calculated_result = Quaternion::from_axis_angle(PI / 2.0, &arr1(&[0.0, 0.0, 1.0]));

        let known_result = Quaternion::new([0.7071067811865476, 0.0, 0.0, 0.7071067811865476]);
        assert_relative_eq!(
            arr1(&known_result.0),
            arr1(&calculated_result.0),
            max_relative = PRECISION,
            epsilon = PRECISION
        );
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
