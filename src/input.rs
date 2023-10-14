use ndarray::Array1;
use serde::Deserialize;
use std::collections::HashMap;
use toml::Table;

use crate::{
    closure::ClosureKind, integralequation::IntegralEquationKind, potential::PotentialKind,
    solver::SolverKind,
};

#[derive(Debug, Deserialize)]
pub struct InputTOML {
    system: System,
    params: Params,
    solvent: Solvent,
    solute: Option<Solute>,
}

#[derive(Debug, Deserialize)]
pub struct System {
    temp: f64,
    #[serde(rename = "kT")]
    boltzmann_internal: f64,
    #[serde(rename = "kU")]
    boltzmann_energy: f64,
    charge_coeff: f64,
    npts: usize,
    radius: f64,
    lam: usize,
}

#[derive(Debug, Deserialize)]
pub struct Params {
    potential: PotentialKind,
    closure: ClosureKind,
    #[serde(rename = "IE")]
    integral_equation: IntegralEquationKind,
    solver: SolverKind,
    depth: Option<usize>,
    picard_damping: f64,
    mdiis_damping: Option<f64>,
    nbasis: Option<usize>,
    itermax: usize,
    tol: f64,
    diel: Option<f64>,
    adbcor: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub enum Solvent {
    Preconverged(String),
    FullData(SolventData),
}

#[derive(Debug, Deserialize)]
pub struct SolventData {
    nsv: usize,
    nspv: usize,
    fields: Vec<Field>,
}

#[derive(Debug, Deserialize)]
pub struct Field {
    dens: f64,
    ns: usize,
    site: HashMap<String, Array1<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct Solute {
    nsu: usize,
    nspv: usize,
    fields: Vec<Field>,
}

#[cfg(test)]
mod test {
    use super::*;

    const TOML_FULL_INPUT: &str = r#"[system]
temp = 298
kT = 1.0
kU = 0.0019872158728366637
charge_coeff = 167101.0
npts = 4086 
radius = 10.24
lam = 1

[params]
potential = "LJ"
closure = "HNC"
IE = "XRISM"
solver = "MDIIS"
depth = 12
picard_damping = 0.5
mdiis_damping = 0.5
itermax = 10000
tol = 1E-5
diel = 78.497
adbcor = 1.5

[solvent]
nsv = 3
nspv = 1

[solute]
nsu = 5
nspu = 1

[solvent.water]
dens = 0.03334
ns = 3
"O" = [
    [78.15, 3.1657, -0.8476000010975563], 
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
]

"H1" = [
    [7.815, 1.1657, 0.4238],
    [1.00000000e+00, 0.00000000e+00, 0.00000000e+00]
] 

"H2" = [
    [7.815, 1.1657, 0.4238],
    [-3.33314000e-01, 9.42816000e-01, 0.00000000e+00]
]

[solute.methane]
dens = 0.0
ns = 5
C1-0 = [ [ 55.05221691240736, 1.6998347542253702, -0.1088,], [ 3.537, 1.423, 0.0,],]
H1-1 = [ [ 7.900546687705287, 1.324766393630111, 0.026699999999999998,], [ 4.089, 2.224, 0.496,],]
H2-2 = [ [ 7.900546687705287, 1.324766393630111, 0.026699999999999998,], [ 4.222, 0.611, -0.254,],]
H3-3 = [ [ 7.900546687705287, 1.324766393630111, 0.026699999999999998,], [ 2.759, 1.049, 0.669,],]
H4-4 = [ [ 7.900546687705287, 1.324766393630111, 0.026699999999999998,], [ 3.077, 1.81, -0.912,],]"#;

    const TOML_PRECONV_INPUT: &str = r#"[system]
temp = 298
kT = 1.0
kU = 0.0019872158728366637
charge_coeff = 167101.0
npts = 4086 
radius = 10.24
lam = 1

[params]
potential = "LJ"
closure = "HNC"
IE = "XRISM"
solver = "MDIIS"
depth = 12
picard_damping = 0.5
mdiis_damping = 0.5
itermax = 10000
tol = 1E-5
diel = 78.497
adbcor = 1.5

[solvent]
preconverged = "path/to/preconverged/binfile"

[solute]
nsu = 5
nspu = 1

[solute.methane]
dens = 0.0
ns = 5
C1-0 = [ [ 55.05221691240736, 1.6998347542253702, -0.1088,], [ 3.537, 1.423, 0.0,],]
H1-1 = [ [ 7.900546687705287, 1.324766393630111, 0.026699999999999998,], [ 4.089, 2.224, 0.496,],]
H2-2 = [ [ 7.900546687705287, 1.324766393630111, 0.026699999999999998,], [ 4.222, 0.611, -0.254,],]
H3-3 = [ [ 7.900546687705287, 1.324766393630111, 0.026699999999999998,], [ 2.759, 1.049, 0.669,],]
H4-4 = [ [ 7.900546687705287, 1.324766393630111, 0.026699999999999998,], [ 3.077, 1.81, -0.912,],]"#;

    #[test]
    fn test_toml_full_data_parse() {
        let input: InputTOML = toml::from_str(TOML_FULL_INPUT).unwrap();
        println!("{:?}", input);
    }

    #[test]
    fn test_toml_preconverged_parse() {
        todo!()
    }
}
