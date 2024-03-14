#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use librism::drivers::rism::{Compress, Verbosity, Write};
    use librism::Calculator;
    use ndarray::Array3;
    use ndarray_stats::QuantileExt;
    use std::path::Path;

    const PRECISION: f64 = 1e-8;

    #[test]
    fn water_isothermal() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let known = 2.0285987231176406;
        let (_, therm, _) = Calculator::new(
            root.join("tests").join("inputs").join("cSPCE.toml"),
            Verbosity::Quiet,
            Compress::NoCompress,
            Write::NoWrite,
        )
        .execute();
        assert_relative_eq!(
            known,
            therm.isothermal_compressibility,
            max_relative = PRECISION,
            epsilon = PRECISION,
        )
    }

    #[test]
    fn argon_rdf_peak() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let known = 2.5250327211581;
        let (sol, _, _) = Calculator::new(
            root.join("tests").join("inputs").join("argon.toml"),
            Verbosity::Quiet,
            Compress::NoCompress,
            Write::NoWrite,
        )
        .execute();

        let gr: Array3<f64> = 1.0 + &sol.uv.unwrap().correlations.hr;

        let rdf_peak: f64 = *gr.max().unwrap();
        assert_relative_eq!(
            known,
            rdf_peak,
            max_relative = PRECISION,
            epsilon = PRECISION,
        )
    }

    #[test]
    fn water_salt_propanol_sfe() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let known = -4.022227508203783;
        let (_, therm, _) = Calculator::new(
            root.join("tests")
                .join("inputs")
                .join("cSPCE_NaCl_propan2ol.toml"),
            Verbosity::Quiet,
            Compress::NoCompress,
            Write::NoWrite,
        )
        .execute();

        let gf = therm.sfe.unwrap().gaussian_fluctuations;
        assert_relative_eq!(known, gf, max_relative = PRECISION, epsilon = PRECISION,)
    }

    #[test]
    fn choloform_pmv() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let known = 116.54335882323188;
        let (_, therm, _) = Calculator::new(
            root.join("tests").join("inputs").join("chloro.toml"),
            Verbosity::Quiet,
            Compress::NoCompress,
            Write::NoWrite,
        )
        .execute();
        assert_relative_eq!(
            known,
            therm.rism_kb_pmv.unwrap(),
            max_relative = PRECISION,
            epsilon = PRECISION,
        )
    }

    #[test]
    fn water_methane_united_atom_pressure() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let known = 0.09448520168155233;
        let (_, therm, _) = Calculator::new(
            root.join("tests")
                .join("inputs")
                .join("cSPCE_XRISM_methane_UA.toml"),
            Verbosity::Quiet,
            Compress::NoCompress,
            Write::NoWrite,
        )
        .execute();
        assert_relative_eq!(
            known,
            therm.pressure.unwrap(),
            max_relative = PRECISION,
            epsilon = PRECISION,
        )
    }

    #[test]
    fn br2_aux_site_tr() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let known = 55.40063292351144;
        let (sol, _, _) = Calculator::new(
            root.join("tests")
                .join("inputs")
                .join("HR1982_Br2_III.toml"),
            Verbosity::Quiet,
            Compress::NoCompress,
            Write::NoWrite,
        )
        .execute();

        let tr_init = sol.vv.correlations.tr[[0, 0, 0]];

        assert_relative_eq!(
            known,
            tr_init,
            max_relative = PRECISION,
            epsilon = PRECISION,
        )
    }
}
