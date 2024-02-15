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
            out.par_mapv_inplace(|x| if x <= d { 0.0 } else { 1e30 });
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
                r: Array1::range(0.5, 8.0, 1.0) * 5.12,
                k: Array1::range(0.5, 8.0, 1.0) * 0.07669903939428206,
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
            2.879_304_053_039_627_9e3,
            -1.525_824_818_710_429_7,
            -7.152_342_312_333_287_4e-2,
            -9.500_933_511_154_720_6e-3,
            -2.103_341_255_723_232_8e-3,
            -6.309_729_358_969_218_5e-4,
            -2.315_833_241_899_503_5e-4,
            -9.813_411_912_377_182_0e-5,
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
            1.025_395_186_315_712_5e9,
            1.135_981_721_847_900_6e8,
            4.065_547_145_434_289_4e7,
            2.056_035_771_141_134_9e7,
            1.229_226_861_672_289_7e7,
            8.108_575_857_716_751_3e6,
            5.703_989_556_533_593_7e6,
            4.197_019_227_849_345_7e6,
        ]);

        assert_relative_eq!(
            known_result,
            calculated_result,
            max_relative = PRECISION * 5.0,
            epsilon = PRECISION * 5.0
        );
    }
}
