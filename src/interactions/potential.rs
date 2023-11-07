use crate::data::configuration::potential::PotentialConfig;
use crate::structure::system::Site;
use errorfunctions::RealErrorFunctions;
use itertools::Itertools;
use ndarray::{s, Array1, Array3};
use pyo3::{prelude::*, types::PyString};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PotentialKind {
    #[serde(rename = "LJ")]
    LennardJones,
    #[serde(rename = "HS")]
    HardSpheres,
    Coulomb,
    NgRenormalisationReal,
    NgRenormalisationFourier,
}

impl<'source> FromPyObject<'source> for PotentialKind {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let str = obj
            .downcast::<PyString>()?
            .to_str()
            .map(ToOwned::to_owned)
            .expect("could not convert string");
        match str.as_str() {
            "LJ" => Ok(PotentialKind::LennardJones),
            "HS" => Ok(PotentialKind::HardSpheres),
            "COU" => Ok(PotentialKind::Coulomb),
            "NGR" => Ok(PotentialKind::NgRenormalisationReal),
            "NGK" => Ok(PotentialKind::NgRenormalisationFourier),
            _ => panic!("not a valid potential"),
        }
    }
}

impl fmt::Display for PotentialKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PotentialKind::LennardJones => write!(f, "Lennard-Jones"),
            PotentialKind::HardSpheres => write!(f, "Hard Spheres"),
            PotentialKind::Coulomb => write!(f, "Coulomb"),
            PotentialKind::NgRenormalisationReal => write!(f, "Ng Renormalisation (Real space)"),
            PotentialKind::NgRenormalisationFourier => {
                write!(f, "Ng Renormalisation (Fourier space)")
            }
        }
    }
}

impl PotentialKind {
    pub fn set(&self) -> fn(&[Site], &[Site], &Array1<f64>, &mut Array3<f64>) {
        match self {
            PotentialKind::LennardJones => lennard_jones,
            PotentialKind::HardSpheres => hard_spheres,
            PotentialKind::Coulomb => coulomb,
            PotentialKind::NgRenormalisationReal => ng_renormalisation_real,
            PotentialKind::NgRenormalisationFourier => ng_renormalisation_fourier,
        }
    }
}

pub struct Potential {
    pub nonbonded: fn(&[Site], &[Site], &Array1<f64>, &mut Array3<f64>),
    pub coulombic: fn(&[Site], &[Site], &Array1<f64>, &mut Array3<f64>),
    pub renormalisation_real: fn(&[Site], &[Site], &Array1<f64>, &mut Array3<f64>),
    pub renormalisation_fourier: fn(&[Site], &[Site], &Array1<f64>, &mut Array3<f64>),
}

impl Potential {
    pub fn new(config: &PotentialConfig) -> Self {
        Potential {
            nonbonded: config.nonbonded.set(),
            coulombic: config.coulombic.set(),
            renormalisation_real: config.renormalisation_real.set(),
            renormalisation_fourier: config.renormalisation_fourier.set(),
        }
    }
}

fn geometric_mean(a: f64, b: f64) -> f64 {
    (a * b).sqrt()
}

fn arithmetic_mean(a: f64, b: f64) -> f64 {
    0.5 * (a + b)
}

fn lorentz_berthelot(eps1: f64, eps2: f64, sig1: f64, sig2: f64) -> (f64, f64) {
    (geometric_mean(eps1, eps2), arithmetic_mean(sig1, sig2))
}

pub fn lennard_jones(
    atoms_a: &[Site],
    atoms_b: &[Site],
    r: &Array1<f64>,
    result: &mut Array3<f64>,
) {
    let atom_pairs = atoms_a
        .iter()
        .enumerate()
        .cartesian_product(atoms_b.iter().enumerate());
    for ((i, site_a), (j, site_b)) in atom_pairs {
        let (eps, sig) = lorentz_berthelot(
            site_a.params[0],
            site_b.params[0],
            site_a.params[1],
            site_b.params[1],
        );
        result.slice_mut(s![.., i, j]).assign({
            let mut ir = sig / r;
            let mut ir6 = ir.view_mut();
            ir6.mapv_inplace(|a| a.powf(6.0));
            let mut ir12 = ir6.to_owned().clone();
            ir12.mapv_inplace(|a| a.powf(2.0));
            &(4.0 * eps * (ir12.to_owned() - ir6.to_owned()))
        })
    }
}

pub fn hard_spheres(atoms_a: &[Site], atoms_b: &[Site], r: &Array1<f64>, result: &mut Array3<f64>) {
    let atom_pairs = atoms_a
        .iter()
        .enumerate()
        .cartesian_product(atoms_b.iter().enumerate());
    for ((i, site_a), (j, site_b)) in atom_pairs {
        let d = arithmetic_mean(site_a.params[0], site_b.params[0]);
        let mut out = r.clone();
        result.slice_mut(s![.., i, j]).assign({
            out.par_mapv_inplace(|x| {
                let new_x;
                if x <= d {
                    new_x = 0.0;
                } else {
                    new_x = 1e30;
                }
                new_x
            });
            &out
        });
    }
}

pub fn coulomb(atoms_a: &[Site], atoms_b: &[Site], r: &Array1<f64>, result: &mut Array3<f64>) {
    let atom_pairs = atoms_a
        .iter()
        .enumerate()
        .cartesian_product(atoms_b.iter().enumerate());
    for ((i, site_a), (j, site_b)) in atom_pairs {
        let q = site_a.params.last().unwrap() * site_b.params.last().unwrap();
        result.slice_mut(s![.., i, j]).assign(&(q / r.clone()));
    }
}

pub fn ng_renormalisation_real(
    atoms_a: &[Site],
    atoms_b: &[Site],
    r: &Array1<f64>,
    result: &mut Array3<f64>,
) {
    let atom_pairs = atoms_a
        .iter()
        .enumerate()
        .cartesian_product(atoms_b.iter().enumerate());
    for ((i, site_a), (j, site_b)) in atom_pairs {
        let q = site_a.params.last().unwrap() * site_b.params.last().unwrap();
        result.slice_mut(s![.., i, j]).assign({
            let mut erf_r = r.clone();
            erf_r.par_mapv_inplace(|x| x.erf());
            &(q * erf_r / r)
        });
    }
}

pub fn ng_renormalisation_fourier(
    atoms_a: &[Site],
    atoms_b: &[Site],
    k: &Array1<f64>,
    result: &mut Array3<f64>,
) {
    let atom_pairs = atoms_a
        .iter()
        .enumerate()
        .cartesian_product(atoms_b.iter().enumerate());
    for ((i, site_a), (j, site_b)) in atom_pairs {
        let q = site_a.params.last().unwrap() * site_b.params.last().unwrap();
        result.slice_mut(s![.., i, j]).assign({
            let mut exp_k = k.clone();
            let mut k2 = k.clone();
            k2.par_mapv_inplace(|x| x.powf(2.0));
            exp_k.par_mapv_inplace(|x| (-1.0 * x.powf(2.0) / 4.0).exp());
            &(q * 4.0 * PI * exp_k / k2)
        });
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{arr1, Array};

    const PRECISION: f64 = f64::EPSILON;

    struct TestData {
        pub oxygen: Vec<Site>,
        pub r: Array1<f64>,
        pub k: Array1<f64>,
    }

    impl TestData {
        fn new() -> Self {
            TestData {
                oxygen: vec![Site {
                    atom_type: String::from("O"),
                    params: vec![78.15, 3.1657, -0.8476000010975563],
                    coords: vec![0.0, 0.0, 0.0],
                }],
                r: Array1::range(0.5, 8 as f64, 1.0) * 5.12,
                k: Array1::range(0.5, 8 as f64, 1.0) * 0.07669903939428206,
            }
        }
    }

    #[test]
    fn test_lennard_jones() {
        let input_data = TestData::new();
        let mut calculated_result = Array3::zeros((8, 1, 1));

        lennard_jones(
            &input_data.oxygen,
            &input_data.oxygen,
            &input_data.r,
            &mut calculated_result,
        );

        let calculated_result = Array::from_iter(calculated_result.iter().cloned());

        let known_result = arr1(&[
            2.8793040530396279e+03,
            -1.5258248187104297e+00,
            -7.1523423123332874e-02,
            -9.5009335111547206e-03,
            -2.1033412557232328e-03,
            -6.3097293589692185e-04,
            -2.3158332418995035e-04,
            -9.8134119123771820e-05,
        ]);

        assert_relative_eq!(
            known_result,
            calculated_result,
            max_relative = PRECISION,
            epsilon = PRECISION
        );
    }

    #[test]
    fn test_coulomb() {
        let input_data = TestData::new();
        let mut calculated_result = Array3::zeros((8, 1, 1));

        coulomb(
            &input_data.oxygen,
            &input_data.oxygen,
            &input_data.r,
            &mut calculated_result,
        );

        calculated_result *= 167101.0;

        let calculated_result = Array::from_iter(calculated_result.iter().cloned());

        let known_result = arr1(&[
            46894.39970025951,
            15631.466566753172,
            9378.879940051902,
            6699.19995717993,
            5210.488855584391,
            4263.127245478137,
            3607.261515404578,
            3126.2933133506344,
        ]);

        assert_relative_eq!(
            known_result,
            calculated_result,
            max_relative = PRECISION,
            epsilon = PRECISION
        );
    }

    #[test]
    fn test_ng_real() {
        let input_data = TestData::new();
        let mut calculated_result = Array3::zeros((8, 1, 1));

        ng_renormalisation_real(
            &input_data.oxygen,
            &input_data.oxygen,
            &input_data.r,
            &mut calculated_result,
        );

        calculated_result *= 167101.0;

        let calculated_result = Array::from_iter(calculated_result.iter().cloned());

        let known_result = arr1(&[
            46880.60510199953,
            15631.466566753172,
            9378.879940051902,
            6699.19995717993,
            5210.488855584391,
            4263.127245478137,
            3607.261515404578,
            3126.2933133506344,
        ]);

        assert_relative_eq!(
            known_result,
            calculated_result,
            max_relative = PRECISION,
            epsilon = PRECISION
        );
    }

    #[test]
    fn test_ng_fourier() {
        let input_data = TestData::new();
        let mut calculated_result = Array3::zeros((8, 1, 1));

        ng_renormalisation_fourier(
            &input_data.oxygen,
            &input_data.oxygen,
            &input_data.k,
            &mut calculated_result,
        );

        calculated_result *= 167101.0;

        let calculated_result = Array::from_iter(calculated_result.iter().cloned());

        let known_result = arr1(&[
            1.0253951863157125e+09,
            1.1359817218479006e+08,
            4.0655471454342894e+07,
            2.0560357711411349e+07,
            1.2292268616722897e+07,
            8.1085758577167513e+06,
            5.7039895565335937e+06,
            4.1970192278493457e+06,
        ]);

        assert_relative_eq!(
            known_result,
            calculated_result,
            max_relative = PRECISION * 5.0,
            epsilon = PRECISION * 5.0
        );
    }
}
