use crate::data::core::DataRs;
use ndarray::{par_azip, Array, Array3};
use pyo3::{prelude::*, types::PyString};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

#[macro_export]
macro_rules! pse_n {
    ($name:ident, $n:expr)  => {
        pub fn $name(problem: &DataRs) -> Array3<f64> {
            let mut out = Array::zeros(problem.correlations.tr.raw_dim());
            let mut t_fac: Array3<f64> = Array::zeros(problem.correlations.tr.raw_dim());
            for i in 0..$n+1{
                t_fac = t_fac + (-problem.system.beta * &problem.interactions.u_sr + &problem.correlations.tr).mapv(|a| a.powf(i as f64)) / factorial(i);
            }
            par_azip!((a in &mut out, &b in &(-problem.system.beta * &problem.interactions.u_sr + &problem.correlations.tr), &c in &problem.correlations.tr, &d in &t_fac)    {
            if b <= 0.0 {
                *a = b.exp() - 1.0 - c
            } else {
                *a = d - 1.0 - c
            }
            });
            out
        }
    };
}

#[macro_export]
macro_rules! pse_derivative_n {
    ($name:ident, $n:expr)  => {
        pub fn $name(problem: &DataRs) -> Array3<f64> {
            let mut out = Array::zeros(problem.correlations.tr.raw_dim());
            let mut t_fac: Array3<f64> = Array::zeros(problem.correlations.tr.raw_dim());
            for i in 2..$n+1 {
                t_fac = t_fac + (-problem.system.beta * &problem.interactions.u_sr + &problem.correlations.tr).mapv(|a| a.powf((i-1) as f64)) * i as f64 / factorial(i);
            }
            par_azip!((a in &mut out, &b in &(-problem.system.beta * &problem.interactions.u_sr + &problem.correlations.tr), &d in &t_fac)    {
            if b <= 0.0 {
                *a = b.exp() - 1.0
            } else {
                *a = d - 1.0
            }
            });
            out
        }
    };
}
pub enum ClosureDerivativeKind {
    HyperNettedChain,
    KovalenkoHirata,
    PercusYevick,
    PartialSeriesExpansion(i8),
}

impl ClosureDerivativeKind {
    pub fn new(closure: &ClosureKind) -> ClosureDerivativeKind {
        match closure {
            ClosureKind::HyperNettedChain => ClosureDerivativeKind::HyperNettedChain,
            ClosureKind::KovalenkoHirata => ClosureDerivativeKind::KovalenkoHirata,
            ClosureKind::PercusYevick => ClosureDerivativeKind::PercusYevick,
            ClosureKind::PartialSeriesExpansion(1) => {
                ClosureDerivativeKind::PartialSeriesExpansion(1)
            }
            ClosureKind::PartialSeriesExpansion(2) => {
                ClosureDerivativeKind::PartialSeriesExpansion(2)
            }
            ClosureKind::PartialSeriesExpansion(3) => {
                ClosureDerivativeKind::PartialSeriesExpansion(3)
            }

            ClosureKind::PartialSeriesExpansion(4) => {
                ClosureDerivativeKind::PartialSeriesExpansion(4)
            }

            ClosureKind::PartialSeriesExpansion(5) => {
                ClosureDerivativeKind::PartialSeriesExpansion(5)
            }

            ClosureKind::PartialSeriesExpansion(6) => {
                ClosureDerivativeKind::PartialSeriesExpansion(6)
            }

            ClosureKind::PartialSeriesExpansion(7) => {
                ClosureDerivativeKind::PartialSeriesExpansion(7)
            }

            _ => panic!("Derivative not implemented for {}", closure),
        }
    }
    pub fn set(&self) -> fn(&DataRs) -> Array3<f64> {
        match self {
            ClosureDerivativeKind::HyperNettedChain => hyper_netted_chain_derivative,
            ClosureDerivativeKind::KovalenkoHirata => kovalenko_hirata_derivative,
            ClosureDerivativeKind::PercusYevick => percus_yevick_derivative,

            ClosureDerivativeKind::PartialSeriesExpansion(1) => {
                partial_series_expansion_derivative_1
            }
            ClosureDerivativeKind::PartialSeriesExpansion(2) => {
                partial_series_expansion_derivative_2
            }
            ClosureDerivativeKind::PartialSeriesExpansion(3) => {
                partial_series_expansion_derivative_3
            }
            ClosureDerivativeKind::PartialSeriesExpansion(4) => {
                partial_series_expansion_derivative_4
            }
            ClosureDerivativeKind::PartialSeriesExpansion(5) => {
                partial_series_expansion_derivative_5
            }
            ClosureDerivativeKind::PartialSeriesExpansion(6) => {
                partial_series_expansion_derivative_6
            }
            ClosureDerivativeKind::PartialSeriesExpansion(7) => {
                partial_series_expansion_derivative_7
            }
            _ => panic!("derivative closure not found"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClosureKind {
    #[serde(rename = "HNC")]
    HyperNettedChain,
    #[serde(rename = "KH")]
    KovalenkoHirata,
    #[serde(rename = "PY")]
    PercusYevick,
    #[serde(rename = "PSE-1")]
    PartialSeriesExpansion1,
    #[serde(rename = "PSE-2")]
    PartialSeriesExpansion2,
    #[serde(rename = "PSE-3")]
    PartialSeriesExpansion3,
    #[serde(rename = "PSE-4")]
    PartialSeriesExpansion4,
    #[serde(rename = "PSE-5")]
    PartialSeriesExpansion5,
    #[serde(rename = "PSE-6")]
    PartialSeriesExpansion6,
    #[serde(rename = "PSE-7")]
    PartialSeriesExpansion7,
}

// impl<'de> Deserialize<'de> for ClosureKind {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: Deserializer<'de>,
//     {
//         let s = String::deserialize(deserializer)?.to_lowercase();
//         let s = s.split("-").collect::<Vec<&str>>();
//         let state = match s[0] {
//             "hnc" => ClosureKind::HyperNettedChain,
//             "kh" => ClosureKind::KovalenkoHirata,
//             "py" => ClosureKind::PercusYevick,
//             "pse" => ClosureKind::PartialSeriesExpansion(
//                 s[1].parse::<i8>().expect("parse PSE n terms into i8"),
//             ),
//             _ => panic!("closure not found"),
//         };
//         Ok(state)
//     }
// }
//
impl<'source> FromPyObject<'source> for ClosureKind {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let str = obj
            .downcast::<PyString>()?
            .to_str()
            .map(ToOwned::to_owned)
            .expect("could not convert string");
        match str.as_str() {
            "HNC" => Ok(ClosureKind::HyperNettedChain),
            "KH" => Ok(ClosureKind::KovalenkoHirata),
            "PSE-1" => Ok(ClosureKind::PartialSeriesExpansion1),
            "PSE-2" => Ok(ClosureKind::PartialSeriesExpansion2),
            "PSE-3" => Ok(ClosureKind::PartialSeriesExpansion3),
            "PSE-4" => Ok(ClosureKind::PartialSeriesExpansion4),
            "PSE-5" => Ok(ClosureKind::PartialSeriesExpansion5),
            "PSE-6" => Ok(ClosureKind::PartialSeriesExpansion6),
            "PSE-7" => Ok(ClosureKind::PartialSeriesExpansion7),
            "PY" => Ok(ClosureKind::PercusYevick),
            _ => panic!("not a valid closure"),
        }
    }
}

impl fmt::Display for ClosureKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ClosureKind::HyperNettedChain => write!(f, "Hyper-Netted Chain"),
            ClosureKind::KovalenkoHirata => write!(f, "Kovalenko-Hirata"),
            ClosureKind::PercusYevick => write!(f, "Percus-Yevick"),
            ClosureKind::PartialSeriesExpansion1 => {
                write!(f, "Partial Series Expansion (1 term)")
            }
            ClosureKind::PartialSeriesExpansion2 => {
                write!(f, "Partial Series Expansion (2 terms)")
            }
            ClosureKind::PartialSeriesExpansion3 => {
                write!(f, "Partial Series Expansion (3 terms)")
            }
            ClosureKind::PartialSeriesExpansion4 => {
                write!(f, "Partial Series Expansion (4 terms)")
            }
            ClosureKind::PartialSeriesExpansion5 => {
                write!(f, "Partial Series Expansion (5 terms)")
            }
            ClosureKind::PartialSeriesExpansion6 => {
                write!(f, "Partial Series Expansion (6 terms)")
            }
            ClosureKind::PartialSeriesExpansion7 => {
                write!(f, "Partial Series Expansion (7 terms)")
            }
        }
    }
}

impl ClosureKind {
    pub fn set(&self) -> fn(&DataRs) -> Array3<f64> {
        match self {
            ClosureKind::HyperNettedChain => hyper_netted_chain,
            ClosureKind::KovalenkoHirata => kovalenko_hirata,
            ClosureKind::PercusYevick => percus_yevick,
            ClosureKind::PartialSeriesExpansion1 => partial_series_expansion_1,
            ClosureKind::PartialSeriesExpansion2 => partial_series_expansion_2,
            ClosureKind::PartialSeriesExpansion3 => partial_series_expansion_3,
            ClosureKind::PartialSeriesExpansion4 => partial_series_expansion_4,
            ClosureKind::PartialSeriesExpansion5 => partial_series_expansion_5,
            ClosureKind::PartialSeriesExpansion6 => partial_series_expansion_6,
            ClosureKind::PartialSeriesExpansion7 => partial_series_expansion_7,
            _ => panic!("not a valid closure"),
        }
    }
}

pub fn hyper_netted_chain(problem: &DataRs) -> Array3<f64> {
    (-problem.system.beta * problem.system.curr_lam * &problem.interactions.u_sr
        + &problem.correlations.tr)
        .mapv(|a| a.exp())
        - 1.0
        - &problem.correlations.tr
}

pub fn hyper_netted_chain_derivative(problem: &DataRs) -> Array3<f64> {
    (-problem.system.beta * problem.system.curr_lam * &problem.interactions.u_sr
        + &problem.correlations.tr)
        .mapv(|a| a.exp())
        - 1.0
}

pub fn kovalenko_hirata(problem: &DataRs) -> Array3<f64> {
    let mut out = Array::zeros(problem.correlations.tr.raw_dim());
    par_azip!((a in &mut out, &b in &(-problem.system.beta * problem.system.curr_lam * &problem.interactions.u_sr + &problem.correlations.tr), &c in &problem.correlations.tr)    {
        if b <= 0.0 {
            *a = b.exp() - 1.0 - c
        } else {
            *a = b - c
        }
    });
    out
}

pub fn kovalenko_hirata_derivative(problem: &DataRs) -> Array3<f64> {
    let mut out = Array::zeros(problem.correlations.tr.raw_dim());
    par_azip!((a in &mut out, &b in &(-problem.system.beta * problem.system.curr_lam * &problem.interactions.u_sr + &problem.correlations.tr))   {
        if b <= 0.0 {
            *a = b.exp() - 1.0
        } else {
            *a = 0.0
        }
    });
    out
}

pub fn percus_yevick(problem: &DataRs) -> Array3<f64> {
    (-problem.system.beta * problem.system.curr_lam * &problem.interactions.u_sr).mapv(|a| a.exp())
        * (1.0 + &problem.correlations.tr)
        - 1.0
        - &problem.correlations.tr
}

pub fn percus_yevick_derivative(problem: &DataRs) -> Array3<f64> {
    (-problem.system.beta * problem.system.curr_lam * &problem.interactions.u_sr).mapv(|a| a.exp())
        - 1.0
}

pub fn factorial(num: u128) -> f64 {
    (1..=num).product::<u128>() as f64
}

pse_n!(partial_series_expansion_1, 1);
pse_n!(partial_series_expansion_2, 2);
pse_n!(partial_series_expansion_3, 3);
pse_n!(partial_series_expansion_4, 4);
pse_n!(partial_series_expansion_5, 5);
pse_n!(partial_series_expansion_6, 6);
pse_n!(partial_series_expansion_7, 7);
pse_derivative_n!(partial_series_expansion_derivative_1, 1);
pse_derivative_n!(partial_series_expansion_derivative_2, 2);
pse_derivative_n!(partial_series_expansion_derivative_3, 3);
pse_derivative_n!(partial_series_expansion_derivative_4, 4);
pse_derivative_n!(partial_series_expansion_derivative_5, 5);
pse_derivative_n!(partial_series_expansion_derivative_6, 6);
pse_derivative_n!(partial_series_expansion_derivative_7, 7);
