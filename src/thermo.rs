use crate::{data::Grid, solution::Solutions, transforms::fourier_bessel_transform_fftw};
use ndarray::{s, Array, Axis, Zip};
use std::f64::consts::PI;

pub struct TDDriver {
    pub solutions: Solutions,
}

impl TDDriver {
    pub fn new(solutions: Solutions) -> Self {
        TDDriver { solutions }
    }

    pub fn isothermal_compressibility(&self) -> f64 {
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

    pub fn kb_partial_molar_volume(&self) -> f64 {
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

    pub fn rism_kb_partial_molar_volume(&self) -> f64 {
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

    pub fn dimensionless_partial_molar_volume(&self) -> f64 {
        let vv = &self.solutions.vv;
        let p_sum = vv
            .data_config
            .solvent_species
            .iter()
            .fold(0.0, |acc, x| acc + x.dens);

        p_sum * self.kb_partial_molar_volume()
    }
}
