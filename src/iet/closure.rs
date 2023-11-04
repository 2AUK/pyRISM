use crate::data::core::DataRs;
use ndarray::{par_azip, Array, Array3};
use pyo3::{prelude::*, types::PyString};
use serde::{Deserialize, Serialize};
use std::fmt;

#[macro_export]
macro_rules! pse_n {
    ($name:ident, $n:expr)  => {
        pub fn $name(problem: &DataRs) -> Array3<f64> {
            let mut out = Array::zeros(problem.correlations.tr.raw_dim());
            let mut t_fac = Array::zeros(problem.correlations.tr.raw_dim());
            for i in 0..$n {
                t_fac += (-problem.system.beta * &problem.interactions.u_sr).mapv(|a| a.powf(i as f64)) / factorial(i);
            }
            par_azip!((a in &mut out, &b in &(-problem.system.beta * &problem.interactions.u_sr + &problem.correlations.tr), &c in &problem.correlations.tr)    {
            if b <= 0.0 {
                *a = b.exp() - 1.0 - c
            } else {
                *a = t_fac - 1.0 - c
            }
            });
            out
        }
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClosureKind {
    #[serde(rename = "HNC")]
    HyperNettedChain,
    #[serde(rename = "KH")]
    KovalenkoHirata,
    #[serde(rename = "PY")]
    PercusYevick,
    #[serde(rename = "PSE")]
    PartialSeriesExpansion(i8),
}

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
            "PSE-1" => Ok(ClosureKind::PartialSeriesExpansion(1)),
            "PSE-2" => Ok(ClosureKind::PartialSeriesExpansion(2)),
            "PSE-3" => Ok(ClosureKind::PartialSeriesExpansion(3)),
            "PSE-4" => Ok(ClosureKind::PartialSeriesExpansion(4)),
            "PSE-5" => Ok(ClosureKind::PartialSeriesExpansion(5)),
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
            ClosureKind::PartialSeriesExpansion(x) => {
                write!(f, "Partial Series Expansion ({} terms)", x)
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
            ClosureKind::PartialSeriesExpansion(_) => partial_series_expansion,
        }
    }
}

pub fn hyper_netted_chain(problem: &DataRs) -> Array3<f64> {
    (-problem.system.beta * &problem.interactions.u_sr + &problem.correlations.tr).mapv(|a| a.exp())
        - 1.0
        - &problem.correlations.tr
}

pub fn kovalenko_hirata(problem: &DataRs) -> Array3<f64> {
    let mut out = Array::zeros(problem.correlations.tr.raw_dim());
    par_azip!((a in &mut out, &b in &(-problem.system.beta * &problem.interactions.u_sr + &problem.correlations.tr), &c in &problem.correlations.tr)    {
        if b <= 0.0 {
            *a = b.exp() - 1.0 - c
        } else {
            *a = b - c
        }
    });
    out
}
pub fn percus_yevick(problem: &DataRs) -> Array3<f64> {
    (-problem.system.beta * &problem.interactions.u_sr).mapv(|a| a.exp())
        * (1.0 + &problem.correlations.tr)
        - 1.0
        - &problem.correlations.tr
}

pub fn partial_series_expansion(_problem: &DataRs) -> Array3<f64> {
    todo!()
}

pub fn factorial(num: u128) -> u128 {
    (1..=num).product()
}
