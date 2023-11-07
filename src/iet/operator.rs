use crate::data::{configuration::operator::OperatorConfig, core::DataRs};
use ndarray::Array3;

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
