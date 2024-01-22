use crate::data::{configuration::operator::OperatorConfig, core::DataRs};
use ndarray::Array3;

use super::closure::ClosureDerivativeKind;

pub struct Operator {
    pub eq: fn(&mut DataRs),
    pub closure: fn(&DataRs) -> Array3<f64>,
    pub closure_der: fn(&DataRs) -> Array3<f64>,
}

impl Operator {
    pub fn new(config: &OperatorConfig) -> Self {
        Operator {
            eq: config.integral_equation.set(),
            closure: config.closure.set(),
            closure_der: ClosureDerivativeKind::new(&config.closure).set(),
        }
    }
}
