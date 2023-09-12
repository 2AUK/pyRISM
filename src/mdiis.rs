use ndarray_linalg::Solve;
use numpy::ndarray::{
    Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, Zip,
};
use numpy::PyArray3;
use pyo3::prelude::*;

#[pyclass]
pub struct MDIIS {
    // arrays that map to python ndarray outputs
    #[pyo3(get, set)]
    pub cr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub tr: Py<PyArray3<f64>>,
    #[pyo3(get, set)]
    pub hk: Py<PyArray3<f64>>,

    // input parameters for solver
    #[pyo3(get, set)]
    pub m: usize,
    #[pyo3(get, set)]
    pub mdiis_damping: f64,
    #[pyo3(get, set)]
    pub picard_damping: f64,

    // store original shape of array
    #[pyo3(get, set)]
    pub npts: usize,
    #[pyo3(get, set)]
    pub ns1: usize,
    #[pyo3(get, set)]
    pub ns2: usize,

    // arrays for MDIIS methods - to only be used in Rust code
    fr: Vec<Array1<f64>>,
    res: Vec<Array1<f64>>,
    rms_res: Vec<Array1<f64>>,
}

impl MDIIS {
    fn picard_step(&mut self, curr: Array3<f64>, prev: Array3<f64>) -> Array3<f64> {

        // calculate difference between current and previous solutions from RISM equation
        let diff = curr.clone() - prev.clone();

        // push current flattened solution into MDIIS array
        self.fr.push(Array::from_iter(curr.clone().into_iter()));

        // push flattened difference into residual array
        self.res.push(Array::from_iter(diff.clone().into_iter()));

        // return Picard iteration step
        prev + self.picard_damping * diff
    }

    fn step_mdiis(&mut self, curr: Array3<f64>, prev: Array3<f64>, gr: &Array1<f64>) -> Array3<f64> {
        let mut a = Array2::zeros((self.m+1, self.m+1));
        let mut b = Array1::zeros(self.m+1);

        // calculate difference between current and previous solutions from RISM equation
        let diff = curr.clone() - prev.clone();

        // push current flattened solution into MDIIS array
        self.fr.push(Array::from_iter(curr.clone().into_iter()));

        // push flattened difference into residual array
        self.res.push(Array::from_iter(diff.clone().into_iter()));

        a[[self.m, self.m]] = 0.0;
        b[[self.m]] = -1.0;

        for i in 0..self.m+1 {
            a[[i, self.m]] = -1.0;
            a[[self.m, i]] = -1.0;
        }

        for i in 0..self.m {
            for j in 0..self.m {
                a[[i, j]] = self.res[i].dot(&self.res[j]);
            }
        }

        let coefficients = a.solve_into(b).expect("could not perform linear solve");

        let mut c_a: Array1<f64> = Array::zeros(self.fr[0].raw_dim());
        let mut min_res: Array1<f64>  = Array::zeros(self.fr[0].raw_dim());
        let denom = (1.0 + gr.mapv(|a| a.powf(2.0))).mapv(f64::sqrt);
        for i in 0..self.m {
            let modified_fr = &self.fr[i] * coefficients[i];
            let modified_res = &self.res[i] * coefficients[i] / &denom;
            c_a += &modified_fr;
            min_res += &modified_res;
        }

        (c_a + self.mdiis_damping * min_res).into_shape((self.npts, self.ns1, self.ns2)).expect("could not reshape array into original shape")
    }
}



#[pymethods]
impl MDIIS {
    #[new]
    fn new(
        cr: Py<PyArray3<f64>>,
        tr: Py<PyArray3<f64>>,
        hk: Py<PyArray3<f64>>,
        m: usize,
        mdiis_damping: f64,
        picard_damping: f64,
        npts: usize,
        ns1: usize,
        ns2: usize,
    ) -> PyResult<Self> {
        Ok(MDIIS {
            cr,
            tr,
            hk,
            m,
            mdiis_damping,
            picard_damping,
            npts,
            ns1,
            ns2,
            fr: Vec::new(),
            res: Vec::new(),
            rms_res: Vec::new(),
        })
    }
}
