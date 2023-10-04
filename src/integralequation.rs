use crate::data::DataRs;
use crate::transforms::fourier_bessel_transform_fftw;
use fftw::plan::*;
use ndarray::{Array, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, Zip};
use ndarray_linalg::Inverse;
use pyo3::{prelude::*, types::PyString};
use std::f64::consts::PI;
use std::fmt;

#[derive(Debug, Clone)]
pub enum IntegralEquationKind {
    XRISM,
    DRISM,
    UV,
}

impl<'source> FromPyObject<'source> for IntegralEquationKind {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let str = obj
            .downcast::<PyString>()?
            .to_str()
            .map(ToOwned::to_owned)
            .expect("could not convert string");
        match str.as_str() {
            "XRISM" => Ok(IntegralEquationKind::XRISM),
            "DRISM" => Ok(IntegralEquationKind::DRISM),
            _ => panic!("not a valid integral equation"),
        }
    }
}

impl fmt::Display for IntegralEquationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntegralEquationKind::XRISM => write!(f, "Extended RISM"),
            IntegralEquationKind::DRISM => write!(f, "Dielectrically Consistent RISM"),
            IntegralEquationKind::UV => write!(f, "Solute-Solvent RISM"),
        }
    }
}

impl IntegralEquationKind {
    pub fn set(&self) -> fn(&mut DataRs, &mut R2RPlan64) {
        match self {
            IntegralEquationKind::XRISM => xrism_vv,
            IntegralEquationKind::DRISM => drism_vv,
            IntegralEquationKind::UV => rism_uv,
        }
    }
}

pub fn rism_uv(_data: &mut DataRs, _plan: &mut R2RPlan64) {
    todo!()
}

pub fn xrism_vv(problem: &mut DataRs, plan: &mut R2RPlan64) {
    let nsv = problem.data_a.sites.len();
    (problem.correlations.hk, problem.correlations.tr) = rism_vv_equation_impl(
        nsv,
        problem.grid.npts,
        problem.grid.rgrid.view(),
        problem.grid.kgrid.view(),
        problem.grid.dr,
        problem.grid.dk,
        problem.correlations.cr.view(),
        problem.data_a.wk.view(),
        problem.data_a.density.view(),
        problem.system.beta,
        problem.interactions.uk_lr.view(),
        problem.interactions.ur_lr.view(),
        Array::zeros((problem.grid.npts, nsv, nsv)).view(),
        plan,
    );
}

pub fn drism_vv(problem: &mut DataRs, plan: &mut R2RPlan64) {
    let nsv = problem.data_a.sites.len();
    (problem.correlations.hk, problem.correlations.tr) = rism_vv_equation_impl(
        nsv,
        problem.grid.npts,
        problem.grid.rgrid.view(),
        problem.grid.kgrid.view(),
        problem.grid.dr,
        problem.grid.dk,
        problem.correlations.cr.view(),
        problem.data_a.wk.view(),
        problem.data_a.density.view(),
        problem.system.beta,
        problem.interactions.uk_lr.view(),
        problem.interactions.ur_lr.view(),
        problem.dielectrics.as_ref().unwrap().chi.view(),
        plan,
    );
}

fn rism_vv_equation_impl(
    nsv: usize,
    _npts: usize,
    r: ArrayView1<f64>,
    k: ArrayView1<f64>,
    dr: f64,
    dk: f64,
    cr: ArrayView3<f64>,
    wk: ArrayView3<f64>,
    p: ArrayView2<f64>,
    b: f64,
    uk_lr: ArrayView3<f64>,
    ur_lr: ArrayView3<f64>,
    chi: ArrayView3<f64>,
    plan: &mut R2RPlan64,
) -> (Array3<f64>, Array3<f64>) {
    // Setting up prefactors for Fourier-Bessel transforms
    let rtok = 2.0 * PI * dr;
    let ktor = dk / (4.0 * PI * PI);

    // Starting FFT Plan
    // let plan = DctPlanner::new().plan_dst4(npts);

    // Setting up arrays used in calculating XRISM equation
    let identity: Array2<f64> = Array::eye(nsv);
    let mut ck = Array::zeros(cr.raw_dim());
    let mut hk = Array::zeros(cr.raw_dim());
    let mut tr = Array::zeros(cr.raw_dim());

    // Transforming c(r) -> c(k)
    Zip::from(cr.lanes(Axis(0)))
        .and(ck.lanes_mut(Axis(0)))
        .for_each(|cr_lane, mut ck_lane| {
            ck_lane.assign(&fourier_bessel_transform_fftw(
                rtok,
                &r,
                &k,
                &cr_lane.to_owned(),
                plan,
            ));
        });
    // Adding long-range component back in
    ck = ck - b * uk_lr.to_owned();

    // Perform integral equation calculation in k-space
    // H = (I - W * C * P)^-1 * (W * C * W)
    Zip::from(hk.outer_iter_mut())
        .and(wk.outer_iter())
        .and(ck.outer_iter())
        .and(chi.outer_iter())
        .par_for_each(|mut hk_matrix, wk_matrix, ck_matrix, chi_matrix| {
            let w_bar = wk_matrix.to_owned() + p.dot(&chi_matrix);
            let iwcp = &identity - w_bar.dot(&ck_matrix.dot(&p));
            let inverted_iwcp = (iwcp).inv().expect("could not invert matrix: {iwcp}");
            let wcw = w_bar.dot(&ck_matrix.dot(&w_bar));
            hk_matrix.assign(&inverted_iwcp.dot(&wcw));
        });

    // Compute t(k) = h(k) - c(k)
    let tk = &hk - ck;

    // Transform t(k) -> t(r)
    Zip::from(tk.lanes(Axis(0)))
        .and(tr.lanes_mut(Axis(0)))
        .for_each(|tk_lane, mut tr_lane| {
            tr_lane.assign(&fourier_bessel_transform_fftw(
                ktor,
                &k,
                &r,
                &tk_lane.to_owned(),
                plan,
            ));
        });

    // removing long-range component
    tr = tr - b * ur_lr.to_owned();

    // return k-space total correlation and r-space indirect correlation functions
    (hk, tr)
}
