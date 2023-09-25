use pyo3::prelude::*;
use crate::closure::ClosureKind;
use crate::integralequation::IntegralEquationKind;
use std::fmt;

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