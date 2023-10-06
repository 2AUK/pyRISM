use crate::closure::ClosureKind;
use crate::data::DataRs;
use crate::integralequation::IntegralEquationKind;
use ndarray::Array3;
use pyo3::prelude::*;
use std::fmt;

pub enum CycleOrder {
    C2T2C,
    T2C2T,
}

#[derive(FromPyObject, Debug, Clone)]
pub struct OperatorConfig {
    pub integral_equation: IntegralEquationKind,
    pub closure: ClosureKind,
}

impl fmt::Display for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Integral Equation: {}\nClosure: {}",
            self.integral_equation, self.closure
        )
    }
}

pub struct Operator {
    pub eq: fn(&mut DataRs),
    pub closure: fn(&DataRs) -> Array3<f64>,
}

impl Operator {
    pub fn new(config: &OperatorConfig) -> Self {
        Operator {
            eq: config.integral_equation.set(),
            closure: config.closure.set(),
        }
    }
}
