use log::info;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;

use crate::{
    data::configuration::{
        Configuration,
        {operator::OperatorConfig, potential::PotentialConfig, problem::ProblemConfig, solver::*},
    },
    iet::closure::ClosureKind,
    iet::integralequation::IntegralEquationKind,
    interactions::potential::PotentialKind,
    solvers::solver::SolverKind,
    structure::system::{Site, Species},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct InputTOMLHandler {
    system: System,
    params: Params,
    solvent: HashMap<String, ProblemInfo>,
    solute: Option<HashMap<String, ProblemInfo>>,
}

impl InputTOMLHandler {
    fn from_toml_string(input: &str) -> Self {
        toml::from_str::<Self>(input).expect("parsed toml string")
    }

    pub fn construct_configuration(fname: &PathBuf) -> Configuration {
        let input_toml_string = fs::read_to_string(fname).expect("input .toml file in string form");
        let input_toml = InputTOMLHandler::from_toml_string(&input_toml_string);
        // Extract operator details
        let operator_config = OperatorConfig {
            integral_equation: input_toml.params.integral_equation,
            closure: input_toml.params.closure,
        };

        // Extract potential details
        let potential_config = PotentialConfig {
            nonbonded: input_toml.params.potential,
            coulombic: PotentialKind::Coulomb,
            renormalisation_real: PotentialKind::NgRenormalisationReal,
            renormalisation_fourier: PotentialKind::NgRenormalisationFourier,
        };

        // Extract solver parameters
        let solver = input_toml.params.solver;
        let picard_damping = input_toml.params.picard_damping;
        let max_iter = input_toml.params.itermax;
        let tolerance = input_toml.params.tol;

        // Getting settings for each solver
        let settings = match solver {
            SolverKind::Picard => SolverSettings {
                picard_damping,
                max_iter,
                tolerance,
                mdiis_settings: None,
                gillan_settings: None,
            },
            SolverKind::Ng => SolverSettings {
                picard_damping,
                max_iter,
                tolerance,
                mdiis_settings: None,
                gillan_settings: None,
            },
            SolverKind::ADIIS => {
                let mdiis_settings = Some(MDIISSettings {
                    depth: input_toml.params.depth.expect("MDIIS depth parameter"),
                    damping: input_toml
                        .params
                        .mdiis_damping
                        .expect("MDIIS mixing parameter"),
                });
                SolverSettings {
                    picard_damping,
                    max_iter,
                    tolerance,
                    mdiis_settings,
                    gillan_settings: None,
                }
            }
            SolverKind::MDIIS => {
                let mdiis_settings = Some(MDIISSettings {
                    depth: input_toml.params.depth.expect("MDIIS depth parameter"),
                    damping: input_toml
                        .params
                        .mdiis_damping
                        .expect("MDIIS mixing parameter"),
                });
                SolverSettings {
                    picard_damping,
                    max_iter,
                    tolerance,
                    mdiis_settings,
                    gillan_settings: None,
                }
            }
            SolverKind::Gillan => {
                let gillan_settings = Some(GillanSettings {
                    nbasis: input_toml
                        .params
                        .nbasis
                        .expect("Gillan basis number parameter"),
                });
                SolverSettings {
                    picard_damping,
                    max_iter,
                    tolerance,
                    mdiis_settings: None,
                    gillan_settings,
                }
            }
            SolverKind::LMV => {
                let gillan_settings = Some(GillanSettings {
                    nbasis: input_toml
                        .params
                        .nbasis
                        .expect("LMV basis number parameter"),
                });
                SolverSettings {
                    picard_damping,
                    max_iter,
                    tolerance,
                    mdiis_settings: None,
                    gillan_settings,
                }
            }
        };
        let solver_config = SolverConfig { solver, settings };

        // Extract solvent data
        let (mut nsv, mut nspv) = (0, 0);
        let mut preconv: Option<PathBuf> = None;
        let mut species: Vec<Species> = Vec::new();
        for (key, value) in input_toml.solvent.iter() {
            match (key.as_str(), value) {
                ("nsv", ProblemInfo::Length(x)) => nsv = *x,
                ("nspv", ProblemInfo::Length(x)) => nspv = *x,
                ("preconverged", ProblemInfo::Preconverged(x)) => preconv = Some(x.into()),
                (name, ProblemInfo::Data(x)) => {
                    let (mut dens, mut ns) = (0.0, 0);
                    let mut sites: Vec<Site> = Vec::new();
                    for (key, value) in x.iter() {
                        match (key.as_str(), value) {
                            ("dens", SpeciesInfo::Num(x)) => dens = *x,
                            ("ns", SpeciesInfo::Num(x)) => ns = *x as usize,
                            (atom_type, SpeciesInfo::SiteInfo(x)) => sites.push(Site {
                                atom_type: String::from(atom_type),
                                params: x[0].clone(),
                                coords: x[1].clone(),
                            }),
                            _ => {
                                panic!("Invalid TOML field {} in [solvent.{}] specified", key, name)
                            }
                        }
                    }
                    species.push(Species {
                        species_name: String::from(name),
                        dens,
                        ns,
                        atom_sites: sites,
                    })
                }
                _ => panic!("Invalid TOML field {} in [solvent] specified", key),
            }
        }
        let atom_sites: Vec<Site> = species.iter().flat_map(|x| x.atom_sites.clone()).collect();

        // Check for solute data and Extract
        let (mut nsu, mut nspu) = (None, None);
        let mut solute_species: Option<Vec<Species>> = None;
        let mut solute_atoms: Option<Vec<Site>> = None;
        match input_toml.solute {
            Some(solute) => {
                let mut _species: Vec<Species> = Vec::new();
                for (key, value) in solute.iter() {
                    match (key.as_str(), value) {
                        ("nsu", ProblemInfo::Length(x)) => nsu = Some(*x),
                        ("nspu", ProblemInfo::Length(x)) => nspu = Some(*x),
                        ("preconverged", ProblemInfo::Preconverged(_)) => (),
                        (name, ProblemInfo::Data(x)) => {
                            let (mut dens, mut ns) = (0.0, 0);
                            let mut sites: Vec<Site> = Vec::new();
                            for (key, value) in x.iter() {
                                match (key.as_str(), value) {
                                    ("dens", SpeciesInfo::Num(x)) => dens = *x,
                                    ("ns", SpeciesInfo::Num(x)) => ns = *x as usize,
                                    (atom_type, SpeciesInfo::SiteInfo(x)) => sites.push(Site {
                                        atom_type: String::from(atom_type),
                                        params: x[0].clone(),
                                        coords: x[1].clone(),
                                    }),
                                    _ => {
                                        panic!(
                                            "Invalid TOML field {} in [solute.{}] specified",
                                            key, name
                                        )
                                    }
                                }
                            }
                            _species.push(Species {
                                species_name: String::from(name),
                                dens,
                                ns,
                                atom_sites: sites,
                            })
                        }
                        _ => panic!("Invalid TOML field {} in [solute] specified", key),
                    }
                }
                solute_species = Some(_species.clone());
                solute_atoms = Some(_species.iter().flat_map(|x| x.atom_sites.clone()).collect());
            }
            _ => info!("No solute data in .toml file"),
        }
        let data_config = ProblemConfig {
            temp: input_toml.system.temp,
            kt: input_toml.system.boltzmann_internal,
            ku: input_toml.system.boltzmann_energy,
            amph: input_toml.system.charge_coeff,
            drism_damping: input_toml.params.adbcor,
            dielec: input_toml.params.diel,
            nsv,
            nsu,
            nspv,
            nspu,
            npts: input_toml.system.npts,
            radius: input_toml.system.radius,
            nlambda: input_toml.system.lam,
            preconverged: preconv,
            solvent_atoms: atom_sites,
            solute_atoms,
            solvent_species: species,
            solute_species,
        };
        Configuration {
            data_config,
            operator_config,
            potential_config,
            solver_config,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
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
    serialize: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProblemInfo {
    Preconverged(String),
    Length(usize),
    Data(BTreeMap<String, SpeciesInfo>),
}

impl ProblemInfo {}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SpeciesInfo {
    Num(f64),
    SiteInfo(Vec<Vec<f64>>),
}

#[cfg(test)]
mod test {
    const _TOML_PARTIAL_INPUT: &str = r#"[system]
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
"dens" = 0.03334
"ns" = 3
"O" = [ 0.0, 0.0, 0.0 ]
"#;

    const _TOML_FULL_INPUT: &str = r#"[system]
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

    const _TOML_PRECONV_INPUT: &str = r#"[system]
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

    // #[test]
    // fn test_toml_full_data_parse() {
    //     let path = PathBuf::from("/home/abdullah/Code/Python/pyRISM/cSPCE_XRISM_methane.toml");
    //     let config = InputTOMLHandler::construct_configuration(&path);
    //     println!("{:?}", config);
    // }
    //
    // #[test]
    // fn test_toml_preconverged_parse() {
    //     let path =
    //         PathBuf::from("/home/abdullah/Code/Python/pyRISM/cSPCE_XRISM_methane_preconv.toml");
    //     let config = InputTOMLHandler::construct_configuration(&path);
    //     println!("{:?}", config);
    // }
}
