use crate::{
    data::{Correlations, Grid},
    solution::Solutions,
    transforms::fourier_bessel_transform_fftw,
};
use ndarray::{s, Array, Array1, Array2, Array3, Axis, NewAxis, Zip};
use std::f64::consts::PI;

pub struct SFEs {
    pub hypernettedchain: f64,
    pub kovalenko_hirata: f64,
    pub gaussian_fluctuations: f64,
    pub partial_wave: f64,
    pub pc_plus: f64,
}

impl SFEs {
    pub fn new(
        beta: f64,
        ku: f64,
        correlations: &Correlations,
        density: &Array2<f64>,
        r: &Array1<f64>,
    ) -> Self {
        let dr = r[[1]] - r[[0]];
        let hnc_sfed =
            hnc_functional_impl(r, density, &correlations.cr, &correlations.hr) / beta * ku;
        let kh_sfed =
            kh_functional_impl(r, density, &correlations.cr, &correlations.hr) / beta * ku;
        let gf_sfed =
            gf_functional_impl(r, density, &correlations.cr, &correlations.hr) / beta * ku;

        println!("HNC: {}", Self::integrate(hnc_sfed, dr));
        println!("KH: {}", Self::integrate(kh_sfed, dr));
        println!("GF: {}", Self::integrate(gf_sfed, dr));
        SFEs {
            hypernettedchain: 0.0,
            kovalenko_hirata: 0.0,
            gaussian_fluctuations: 0.0,
            partial_wave: 0.0,
            pc_plus: 0.0,
        }
    }

    pub fn integrate(func: Array1<f64>, dr: f64) -> f64 {
        func.sum() * dr
    }
}

pub struct Densities {
    pub hypernettedchain: Array1<f64>,
    pub kovalenko_hirate: Array1<f64>,
    pub gaussian_fluctuations: Array1<f64>,
    pub partial_wave: Array1<f64>,
    pub partial_molar_volume: Array1<f64>,
}

pub struct Thermodynamics {
    pub sfe: SFEs,
    pub sfed: Densities,
    pub isothermal_compressibility: f64,
    pub molecular_kb_pmv: f64,
    pub rism_kb_pmv: f64,
    pub total_density: f64,
}

pub struct TDDriver {
    // pub thermodynamics: Thermodynamics,
    pub solutions: Solutions,
}

impl TDDriver {
    pub fn new(solutions: Solutions) -> Self {
        TDDriver { solutions }
    }

    fn isothermal_compressibility(&self) -> f64 {
        let vv = &self.solutions.vv;
        let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
        let rtok = 2.0 * PI * grid.dr;
        let mut ck = Array::zeros(vv.correlations.cr.raw_dim());

        Zip::from(vv.correlations.cr.lanes(Axis(0)))
            .and(ck.lanes_mut(Axis(0)))
            .par_for_each(|cr_lane, mut ck_lane| {
                ck_lane.assign(&fourier_bessel_transform_fftw(
                    rtok,
                    &grid.rgrid.view(),
                    &grid.kgrid.view(),
                    &cr_lane.to_owned(),
                ));
            });

        let c_sum = ck.slice(s![0, .., ..]).sum();
        let p_sum = vv
            .data_config
            .solvent_species
            .iter()
            .fold(0.0, |acc, x| acc + x.dens);

        1.0 / (p_sum * (1.0 - p_sum * c_sum))
    }

    fn kb_partial_molar_volume(&self) -> f64 {
        let vv = &self.solutions.vv;
        let uv = &self.solutions.uv.as_ref().unwrap();

        let p_sum = vv
            .data_config
            .solvent_species
            .iter()
            .fold(0.0, |acc, x| acc + x.dens);

        let n = uv.data_config.nsu.unwrap() as f64;

        let hkvv_sum = vv.correlations.hk.slice(s![0, .., ..]).sum();
        let hkuv_sum = uv.correlations.hk.slice(s![0, .., ..]).sum();

        (1.0 / p_sum) + (hkvv_sum - hkuv_sum) / n
    }

    fn rism_kb_partial_molar_volume(&self) -> f64 {
        let vv = &self.solutions.vv;
        let uv = &self.solutions.uv.as_ref().unwrap();
        let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
        let rtok = 2.0 * PI * grid.dr;
        let _inv_beta = vv.data_config.temp * vv.data_config.kt;
        let mut ck = Array::zeros(uv.correlations.cr.raw_dim());

        Zip::from(uv.correlations.cr.lanes(Axis(0)))
            .and(ck.lanes_mut(Axis(0)))
            .par_for_each(|cr_lane, mut ck_lane| {
                ck_lane.assign(&fourier_bessel_transform_fftw(
                    rtok,
                    &grid.rgrid.view(),
                    &grid.kgrid.view(),
                    &cr_lane.to_owned(),
                ));
            });

        let c_sum = ck.slice(s![0, .., ..]).sum();
        let p_sum = vv
            .data_config
            .solvent_species
            .iter()
            .fold(0.0, |acc, x| acc + x.dens);
        let compressibility = self.isothermal_compressibility();
        compressibility * (1.0 - p_sum * c_sum)
    }

    fn total_density(&self) -> f64 {
        let vv = &self.solutions.vv;
        vv.data_config
            .solvent_species
            .iter()
            .fold(0.0, |acc, x| acc + x.dens)
    }
}

fn hnc_functional_impl(
    r: &Array1<f64>,
    density: &Array2<f64>,
    cr: &Array3<f64>,
    hr: &Array3<f64>,
) -> Array1<f64> {
    let tr = hr - cr;
    let mut _out: Array3<f64> = Array::zeros(tr.raw_dim());
    //let r = r.slice(s![.., NewAxis, NewAxis]).to_owned();
    let mut r = r.broadcast((1, 1, r.len())).unwrap();
    r.swap_axes(0, 2);
    Zip::from(_out.outer_iter_mut())
        .and(tr.outer_iter())
        .and(cr.outer_iter())
        .and(hr.outer_iter())
        .and(r.outer_iter())
        .for_each(|mut o, t, c, h, ri| {
            let r2 = &ri * &ri;
            let integrand = 0.5 * &t * &h - &c;
            o.assign(&(4.0 * PI * &(r2 * integrand).dot(density)));
        });
    _out.sum_axis(Axis(2)).sum_axis(Axis(1))
}

fn kh_functional_impl(
    r: &Array1<f64>,
    density: &Array2<f64>,
    cr: &Array3<f64>,
    hr: &Array3<f64>,
) -> Array1<f64> {
    let tr = hr - cr;
    let mut _out: Array3<f64> = Array::zeros(tr.raw_dim());
    //let r = r.slice(s![.., NewAxis, NewAxis]).to_owned();
    let mut r = r.broadcast((1, 1, r.len())).unwrap();
    r.swap_axes(0, 2);
    Zip::from(_out.outer_iter_mut())
        .and(tr.outer_iter())
        .and(cr.outer_iter())
        .and(hr.outer_iter())
        .and(r.outer_iter())
        .for_each(|mut o, t, c, h, ri| {
            let r2 = &ri.dot(&ri);
            let integrand = 0.5 * &h * &h * heaviside(&(-h.to_owned())) - (0.5 * &h * &c) - &c;
            o.assign(&(4.0 * PI * &(r2 * integrand).dot(density)));
        });
    _out.sum_axis(Axis(2)).sum_axis(Axis(1))
}

fn gf_functional_impl(
    r: &Array1<f64>,
    density: &Array2<f64>,
    cr: &Array3<f64>,
    hr: &Array3<f64>,
) -> Array1<f64> {
    let tr = hr - cr;
    let mut _out: Array3<f64> = Array::zeros(tr.raw_dim());
    //let r = r.slice(s![.., NewAxis, NewAxis]).to_owned();
    let mut r = r.broadcast((1, 1, r.len())).unwrap();
    r.swap_axes(0, 2);
    Zip::from(_out.outer_iter_mut())
        .and(tr.outer_iter())
        .and(cr.outer_iter())
        .and(hr.outer_iter())
        .and(r.outer_iter())
        .for_each(|mut o, t, c, h, ri| {
            let r2 = &ri * &ri;
            let integrand = 0.5 * &c * &h + &c;
            o.assign(&(-4.0 * PI * &(r2 * integrand).dot(density)));
        });
    _out.sum_axis(Axis(2)).sum_axis(Axis(1))
}

fn pw_functional_impl(
    r: &Array1<f64>,
    density: &Array2<f64>,
    cr: &Array3<f64>,
    hr: &Array3<f64>,
) -> Array1<f64> {
    let tr = hr - cr;
    let mut _out: Array3<f64> = Array::zeros(tr.raw_dim());
    //let r = r.slice(s![.., NewAxis, NewAxis]).to_owned();
    let mut r = r.broadcast((1, 1, r.len())).unwrap();
    r.swap_axes(0, 2);
    Zip::from(_out.outer_iter_mut())
        .and(tr.outer_iter())
        .and(cr.outer_iter())
        .and(hr.outer_iter())
        .and(r.outer_iter())
        .for_each(|mut o, t, c, h, ri| {
            let r2 = &ri * &ri;
            let integrand = 0.5 * &c * &h + &c;
            o.assign(&(4.0 * PI * &(r2 * integrand).dot(density)));
        });
    _out.sum_axis(Axis(2)).sum_axis(Axis(1))
}

fn heaviside(arr: &Array2<f64>) -> Array2<f64> {
    Array::from_shape_vec(
        arr.raw_dim(),
        arr.iter()
            .map(|x| if *x < 0.0 { 0.0 } else { 1.0 })
            .collect(),
    )
    .expect("heaviside function from input array")
}
