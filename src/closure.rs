use crate::data::DataRs;
use ndarray::Array3;
use pyo3::{prelude::*, types::PyString};
use std::fmt;

#[derive(Debug, Clone)]
pub enum ClosureKind {
    HyperNettedChain,
    KovalenkoHirata,
    PercusYevick,
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

// pub fn hyper_netted_chain(b: f64, u: ArrayView3<f64>, t: ArrayView3<f64>) -> Array3<f64> {
//     (-b * u.to_owned() + t).mapv(|a| a.exp()) - 1.0 - t
// }

pub fn hyper_netted_chain(data: &DataRs) -> Array3<f64> {
    (-data.beta * &data.u_sr + &data.tr).mapv(|a| a.exp()) - 1.0 - &data.tr
}
