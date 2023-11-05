use crate::{
    data::core::Correlations,
    data::solution::Solutions,
    grids::{radial_grid::Grid, transforms::fourier_bessel_transform_fftw},
};
use ndarray::{s, Array, Array1, Array2, Array3, Axis, Zip};
use ndarray_linalg::Inverse;
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
        w_u: &Array3<f64>,
        w_v: &Array3<f64>,
        density: &Array2<f64>,
        r: &Array1<f64>,
        k: &Array1<f64>,
    ) -> Self {
        let dr = r[[1]] - r[[0]];
        let hnc_sfed =
            hnc_functional_impl(r, density, &correlations.cr, &correlations.hr) / beta * ku;
        let kh_sfed =
            kh_functional_impl(r, density, &correlations.cr, &correlations.hr) / beta * ku;
        let gf_sfed =
            gf_functional_impl(r, density, &correlations.cr, &correlations.hr) / beta * ku;
        let pw_sfed = pw_functional_impl(
            r,
            k,
            density,
            &correlations.cr,
            &correlations.hr,
            &correlations.hk,
            w_u,
            w_v,
        ) / beta
            * ku;

        println!("HNC: {}", Self::integrate(hnc_sfed, dr));
        println!("KH: {}", Self::integrate(kh_sfed, dr));
        println!("GF: {}", Self::integrate(gf_sfed, dr));
        println!("PW: {}", Self::integrate(pw_sfed, dr));

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

    pub fn print_thermo(&self) {
        println!(
            "Isothermal Compressibility: {}",
            self.isothermal_compressibility()
        );
        println!(
            "Molecular KB theory PMV: {} (A^3)",
            self.kb_partial_molar_volume()
        );
        println!(
            "Molecular KB theory PMV: {} (cm^3 / mol)",
            self.kb_partial_molar_volume() / 1e24 * 6.022e23
        );
        println!(
            "RISM KB theory PMV: {} (A^3)",
            self.rism_kb_partial_molar_volume()
        );
        println!(
            "RISM KB theory PMV: {} (cm^3 / mol)",
            self.rism_kb_partial_molar_volume() / 1e24 * 6.022e23
        );
        println!(
            "RISM KB theory Excess Volume: {} (A^3)",
            self.rism_kb_partial_molar_volume() - self.isothermal_compressibility()
        );
        println!(
            "RISM KB theory Excess Volume: {} (cm^3 / mol)",
            (self.rism_kb_partial_molar_volume() - self.isothermal_compressibility()) / 1e24
                * 6.022e23
        );
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
    k: &Array1<f64>,
    density: &Array2<f64>,
    cr: &Array3<f64>,
    hr: &Array3<f64>,
    hk: &Array3<f64>,
    w_u: &Array3<f64>,
    w_v: &Array3<f64>,
) -> Array1<f64> {
    let tr = hr - cr;
    let dk = k[[1]] - k[[0]];
    let ktor = dk / (4.0 * PI * PI);
    let mut h_bar_uv_k: Array3<f64> = Array::zeros(tr.raw_dim());
    let mut h_bar_uv_r: Array3<f64> = Array::zeros(tr.raw_dim());
    let mut _out: Array3<f64> = Array::zeros(tr.raw_dim());

    Zip::from(h_bar_uv_k.outer_iter_mut())
        .and(hk.outer_iter())
        .and(w_u.outer_iter())
        .and(w_v.outer_iter())
        .par_for_each(|mut h_out, h, wu, wv| {
            let wv_inv = wv.inv().expect("inverted solvent intramolecular matrix");
            let wu_inv = wu.inv().expect("inverted solute intramolecular matrix");
            h_out.assign(&wu.dot(&h).dot(&wv).dot(density));
        });

    Zip::from(h_bar_uv_k.lanes(Axis(0)))
        .and(h_bar_uv_r.lanes_mut(Axis(0)))
        .par_for_each(|cr_lane, mut ck_lane| {
            ck_lane.assign(&fourier_bessel_transform_fftw(
                ktor,
                &k.view(),
                &r.view(),
                &cr_lane.to_owned(),
            ));
        });

    let mut r = r.broadcast((1, 1, r.len())).unwrap();
    r.swap_axes(0, 2);

    Zip::from(_out.outer_iter_mut())
        .and(h_bar_uv_r.outer_iter())
        .and(cr.outer_iter())
        .and(hr.outer_iter())
        .and(r.outer_iter())
        .for_each(|mut o, h_bar, c, h, ri| {
            let r2 = &ri * &ri;
            let integrand = &c + (0.5 * &c * &h) - (0.5 * &h_bar * &h);
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