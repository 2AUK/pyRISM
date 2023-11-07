use crate::{
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
    pub fn new(densities: &Densities, pressure: f64, pmv: f64, dr: f64) -> Self {
        SFEs {
            hypernettedchain: Self::integrate(&densities.hypernettedchain, dr),
            kovalenko_hirata: Self::integrate(&densities.kovalenko_hirata, dr),
            gaussian_fluctuations: Self::integrate(&densities.gaussian_fluctuations, dr),
            partial_wave: Self::integrate(&densities.partial_wave, dr),
            pc_plus: pressure * pmv,
        }
    }

    pub fn integrate(func: &Array1<f64>, dr: f64) -> f64 {
        func.sum() * dr
    }
}

impl std::fmt::Display for SFEs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Solvation Free Energies (kcal/mol):
\tHNC:                          {}
\tKH:                           {}
\tGF:                           {}
\tPW:                           {}
\tPC+:                          {}
\tPW - PC+                      {}",
            self.hypernettedchain,
            self.kovalenko_hirata,
            self.gaussian_fluctuations,
            self.partial_wave,
            self.pc_plus,
            self.partial_wave - self.pc_plus,
        )
    }
}

pub struct Densities {
    pub hypernettedchain: Array1<f64>,
    pub kovalenko_hirata: Array1<f64>,
    pub gaussian_fluctuations: Array1<f64>,
    pub partial_wave: Array1<f64>,
    partial_molar_volume: Array1<f64>,
}

impl Densities {
    pub fn new(
        solutions: &Solutions,
        wv: &Array3<f64>,
        wu: &Array3<f64>,
        pmv_density: Array1<f64>,
    ) -> Self {
        let uv = solutions.uv.as_ref().unwrap();
        let grid = Grid::new(
            solutions.config.data_config.npts,
            solutions.config.data_config.radius,
        );
        let temp = solutions.config.data_config.temp;
        let kt = solutions.config.data_config.kt;
        let ku = solutions.config.data_config.ku;
        let beta = 1.0 / kt / temp;

        let density = {
            let mut dens_vec: Vec<f64> = Vec::new();
            for i in solutions
                .config
                .data_config
                .solvent_species
                .clone()
                .into_iter()
            {
                for _j in i.atom_sites {
                    dens_vec.push(i.dens);
                }
            }
            &Array2::from_diag(&Array::from_vec(dens_vec))
        };

        let r = &grid.rgrid;
        let k = &grid.kgrid;
        let cr = &uv.correlations.cr;
        let hr = &uv.correlations.hr;
        let hk = &uv.correlations.hk;
        let hnc_density = hnc_functional_impl(r, density, cr, hr) / beta * ku;
        let gf_density = gf_functional_impl(r, density, cr, hr) / beta * ku;
        let kh_density = kh_functional_impl(r, density, cr, hr) / beta * ku;
        let pw_density = pw_functional_impl(r, k, density, cr, hr, hk, wu, wv) / beta * ku;

        Densities {
            hypernettedchain: hnc_density,
            kovalenko_hirata: kh_density,
            gaussian_fluctuations: gf_density,
            partial_wave: pw_density,
            partial_molar_volume: pmv_density,
        }
    }
}

pub struct Thermodynamics {
    pub total_density: f64,
    pub temperature: f64,
    pub isothermal_compressibility: f64,
    pub sfe: Option<SFEs>,
    pub sfed: Option<Densities>,
    pub molecular_kb_pmv: Option<f64>,
    pub rism_kb_pmv: Option<f64>,
    pub pressure: Option<f64>,
}

impl std::fmt::Display for Thermodynamics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.sfe {
            Some(ref sfe) => {
                write!(
                    f,
                    "Thermodynamics:
\tTemperature:                  {} K
\tTotal Density:                {} (1/A^3)
\tIsothermal Compressibility:   {}
\tPressure:                     {}
\tMolecular KB PMV:             {} A^3
\t                              {} cm^3/mol
\tRISM KB PMV:                  {} A^3
\t                              {} cm^3/mol
{}",
                    self.temperature,
                    self.total_density,
                    self.isothermal_compressibility,
                    self.pressure.unwrap(),
                    self.molecular_kb_pmv.unwrap(),
                    self.molecular_kb_pmv.unwrap() / 1e24 * 6.022e23,
                    self.rism_kb_pmv.unwrap(),
                    self.rism_kb_pmv.unwrap() / 1e24 * 6.022e23,
                    sfe,
                )
            }
            None => {
                write!(
                    f,
                    "Thermodynamics:
\tTemperature:                  {} K
\tTotal Density:                {} (1/A^3)
\tIsothermal Compressibility:   {}",
                    self.temperature, self.total_density, self.isothermal_compressibility,
                )
            }
        }
    }
}
pub struct TDDriver<'a> {
    pub solutions: &'a Solutions,
    pub(crate) wv: Array3<f64>,
    pub(crate) wu: Option<Array3<f64>>,
}

impl<'a> TDDriver<'a> {
    pub fn new(solutions: &'a Solutions, wv: Array3<f64>, wu: Option<Array3<f64>>) -> Self {
        TDDriver { solutions, wv, wu }
    }

    pub fn execute(&self) -> Thermodynamics {
        let grid = Grid::new(
            self.solutions.config.data_config.npts,
            self.solutions.config.data_config.radius,
        );
        match self.solutions.uv {
            Some(_) => {
                let isothermal_compressibility = self.isothermal_compressibility();
                let molecular_kb_pmv = self.kb_partial_molar_volume();
                let rism_kb_pmv = self.rism_kb_partial_molar_volume();
                let total_density = self.total_density();
                let sfed = Densities::new(
                    &self.solutions,
                    &self.wv,
                    self.wu.as_ref().unwrap(),
                    self.rism_kb_partial_molar_volume_density_2(),
                );
                let pressure = self.pressure();
                let sfe = SFEs::new(&sfed, pressure, rism_kb_pmv, grid.dr);
                let temperature = self.solutions.config.data_config.temp;
                //
                // println!(
                //     "1: {}\n2: {}\n3: {}\n4: {}",
                //     SFEs::integrate(&self.rism_kb_partial_molar_volume_density(), grid.dr),
                //     SFEs::integrate(&self.rism_kb_partial_molar_volume_density_2(), grid.dr),
                //     SFEs::integrate(
                //         &(&self.rism_kb_partial_molar_volume_density()
                //             - self.isothermal_compressibility()),
                //         grid.dr
                //     ) + self.isothermal_compressibility(),
                //     SFEs::integrate(
                //         &(&self.rism_kb_partial_molar_volume_density_2()
                //             - self.isothermal_compressibility_density()),
                //         grid.dr
                //     ) + SFEs::integrate(
                //         &(&self.isothermal_compressibility_density()
                //             - (1.0 / self.total_density())),
                //         grid.dr
                //     ) + (1.0 / self.total_density()),
                // );

                Thermodynamics {
                    temperature,
                    total_density,
                    isothermal_compressibility,
                    sfe: Some(sfe),
                    sfed: Some(sfed),
                    molecular_kb_pmv: Some(molecular_kb_pmv),
                    rism_kb_pmv: Some(rism_kb_pmv),
                    pressure: Some(pressure),
                }
            }
            None => {
                let isothermal_compressibility = self.isothermal_compressibility();
                let total_density = self.total_density();
                let temperature = self.solutions.config.data_config.temp;

                Thermodynamics {
                    temperature,
                    total_density,
                    isothermal_compressibility,
                    sfe: None,
                    sfed: None,
                    molecular_kb_pmv: None,
                    rism_kb_pmv: None,
                    pressure: None,
                }
            }
        }
    }

    fn pressure(&self) -> f64 {
        let grid = Grid::new(
            self.solutions.config.data_config.npts,
            self.solutions.config.data_config.radius,
        );
        let uv = self.solutions.uv.as_ref().unwrap();
        let ku = self.solutions.config.data_config.ku;
        let nu = self.solutions.config.data_config.nsu.unwrap();
        let temp = self.solutions.config.data_config.temp;
        let mut ck = Array::zeros(uv.correlations.cr.raw_dim());
        let rtok = 2.0 * PI * grid.dr;

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
        let p = self.total_density();
        let initial_term = ((nu as f64 + 1.0) / 2.0) * ku * temp * p;
        let ck_direct = p * p * ck.slice(s![0, .., ..]).sum();
        let pressure = initial_term - (ku * temp) / 2.0 * ck_direct;
        let ideal_pressure = p * ku * temp;
        pressure - ideal_pressure
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

    fn isothermal_compressibility_density(&self) -> Array1<f64> {
        let vv = &self.solutions.vv;
        let cr = &vv.correlations.cr;
        let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
        let r = &grid.rgrid;
        let site_summed_cr = r * r * cr.sum_axis(Axis(2)).sum_axis(Axis(1)) * 4.0 * PI;
        let p_sum = vv
            .data_config
            .solvent_species
            .iter()
            .fold(0.0, |acc, x| acc + x.dens);

        1.0 / (p_sum * (1.0 - p_sum * site_summed_cr))
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

    fn rism_kb_partial_molar_volume_density(&self) -> Array1<f64> {
        let vv = &self.solutions.vv;
        let uv = &self.solutions.uv.as_ref().unwrap();
        let cr = &uv.correlations.cr;
        let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
        let r = &grid.rgrid;
        let site_summed_cr = cr.sum_axis(Axis(2)).sum_axis(Axis(1));
        let integrand = r * r * site_summed_cr * 4.0 * PI;
        let compressibility = self.isothermal_compressibility();
        let p_sum = self.total_density();

        compressibility * (1.0 - p_sum * integrand)
    }

    fn rism_kb_partial_molar_volume_density_2(&self) -> Array1<f64> {
        let vv = &self.solutions.vv;
        let uv = &self.solutions.uv.as_ref().unwrap();
        let cr = &uv.correlations.cr;
        let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
        let r = &grid.rgrid;
        let site_summed_cr = cr.sum_axis(Axis(2)).sum_axis(Axis(1));
        let integrand = r * r * site_summed_cr * 4.0 * PI;
        let compressibility = self.isothermal_compressibility_density();
        let p_sum = self.total_density();

        compressibility * (1.0 - p_sum * integrand)
        //1.0 - integrand
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
            h_out.assign(&wu_inv.dot(&h).dot(&wv_inv).dot(density));
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
            o.assign(&(-4.0 * PI * &(r2 * integrand).dot(density)));
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
