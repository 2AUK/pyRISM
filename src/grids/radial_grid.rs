use ndarray::Array1;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Grid {
    pub npts: usize,
    pub radius: f64,
    pub dr: f64,
    pub dk: f64,
    pub rgrid: Array1<f64>,
    pub kgrid: Array1<f64>,
}

impl Grid {
    pub fn new(npts: usize, radius: f64) -> Self {
        let dr = radius / npts as f64;
        let dk = 2.0 * PI / (2.0 * npts as f64 * dr);
        Grid {
            npts,
            radius,
            dr,
            dk,
            rgrid: Array1::range(0.5, npts as f64, 1.0) * dr,
            kgrid: Array1::range(0.5, npts as f64, 1.0) * dk,
        }
    }
}
