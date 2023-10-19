use ndarray::Array;

use crate::solution::Solutions;

pub struct TDDriver {
    pub solutions: Solutions,
}

impl TDDriver {
    pub fn new(solutions: Solutions) -> Self {
        TDDriver { solutions }
    }
    pub fn isothermal_compressibility(&self) -> f64 {
        let vv = &self.solutions.vv;
        let ck = Array::zeros(vv.correlations.cr.raw_dim());

        todo!()
    }
}
