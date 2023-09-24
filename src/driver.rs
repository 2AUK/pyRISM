use crate::data::{self, DataRs};
use crate::mdiis::MDIIS;
use pyo3::{prelude::*, types::PyString};
use std::fmt;

#[derive(FromPyObject, Debug, Clone)]
pub struct Site {
    pub atom_type: String,
    pub params: Vec<f64>,
    pub coords: Vec<f64>,
}

#[derive(FromPyObject, Debug, Clone)]
pub struct Species {
    pub species_name: String,
    pub dens: f64,
    pub ns: usize,
    pub atom_sites: Vec<Site>,
}

#[derive(FromPyObject, Debug, Clone)]
pub struct DataConfig {
    pub temp: f64,
    pub kt: f64,
    pub ku: f64,
    pub amph: f64,
    pub nsv: usize,
    pub nsu: Option<usize>,
    pub nspv: usize,
    pub nspu: Option<usize>,
    pub npts: usize,
    pub radius: f64,
    pub nlambda: usize,
    pub atoms: Vec<Site>,
    pub solvent_species: Vec<Species>,
    pub solute_species: Option<Vec<Species>>,
}

#[derive(FromPyObject, Debug, Clone)]
pub struct PotentialConfig {
    pub nonbonded: PotentialKind,
    pub coulombic: PotentialKind,
    pub renormalisation: PotentialKind,
}

impl fmt::Display for PotentialConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Potential: {}", self.nonbonded)
    }
}

#[derive(Debug, Clone)]
pub enum SolverKind {
    Picard,
    Ng,
    MDIIS,
    Gillan,
}

impl fmt::Display for SolverKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SolverKind::Picard => write!(f, "Picard"),
            SolverKind::Ng => write!(f, "Ng"),
            SolverKind::MDIIS => write!(f, "MDIIS"),
            SolverKind::Gillan => write!(f, "Gillan"),
        }
    }
}

impl<'source> FromPyObject<'source> for SolverKind {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let str = obj
            .downcast::<PyString>()?
            .to_str()
            .map(ToOwned::to_owned)
            .expect("could not convert string");
        match str.as_str() {
            "Picard" => Ok(SolverKind::Picard),
            "Ng" => Ok(SolverKind::Ng),
            "MDIIS" => Ok(SolverKind::MDIIS),
            "Gillan" => Ok(SolverKind::Gillan),
            _ => panic!("not a valid solver"),
        }
    }
}

#[derive(FromPyObject, Debug, Clone)]
pub struct SolverConfig {
    pub solver: SolverKind,
}

impl fmt::Display for SolverConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Solver: {}", self.solver)
    }
}

#[derive(FromPyObject, Debug, Clone)]
pub struct OperatorConfig {
    pub integral_equation: IntegralEquationKind,
    pub closure: ClosureKind,
}

impl fmt::Display for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Integral Equation: {}\nClosure: {}",
            self.integral_equation, self.closure
        )
    }
}

#[derive(Debug, Clone)]
pub enum PotentialKind {
    LennardJones,
    HardSpheres,
    Coulomb,
    NgRenormalisation,
}

impl<'source> FromPyObject<'source> for PotentialKind {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let str = obj
            .downcast::<PyString>()?
            .to_str()
            .map(ToOwned::to_owned)
            .expect("could not convert string");
        match str.as_str() {
            "LJ" => Ok(PotentialKind::LennardJones),
            "HS" => Ok(PotentialKind::HardSpheres),
            "COU" => Ok(PotentialKind::Coulomb),
            "NG" => Ok(PotentialKind::NgRenormalisation),
            _ => panic!("not a valid potential"),
        }
    }
}

impl fmt::Display for PotentialKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PotentialKind::LennardJones => write!(f, "Lennard-Jones"),
            PotentialKind::HardSpheres => write!(f, "Hard Spheres"),
            PotentialKind::Coulomb => write!(f, "Coulomb"),
            PotentialKind::NgRenormalisation => write!(f, "Ng Renormalisation"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ClosureKind {
    HyperNettedChain,
    KovalenkoHirata,
    PercusYevick,
    PartialSeriesExpansion(i8),
}

impl<'source> FromPyObject<'source> for ClosureKind {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let str = obj
            .downcast::<PyString>()?
            .to_str()
            .map(ToOwned::to_owned)
            .expect("could not convert string");
        match str.as_str() {
            "HNC" => Ok(ClosureKind::HyperNettedChain),
            "KH" => Ok(ClosureKind::KovalenkoHirata),
            "PSE-1" => Ok(ClosureKind::PartialSeriesExpansion(1)),
            "PSE-2" => Ok(ClosureKind::PartialSeriesExpansion(2)),
            "PSE-3" => Ok(ClosureKind::PartialSeriesExpansion(3)),
            "PSE-4" => Ok(ClosureKind::PartialSeriesExpansion(4)),
            "PSE-5" => Ok(ClosureKind::PartialSeriesExpansion(5)),
            "PY" => Ok(ClosureKind::PercusYevick),
            _ => panic!("not a valid closure"),
        }
    }
}

impl fmt::Display for ClosureKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ClosureKind::HyperNettedChain => write!(f, "Hyper-Netted Chain"),
            ClosureKind::KovalenkoHirata => write!(f, "Kovalenko-Hirata"),
            ClosureKind::PercusYevick => write!(f, "Percus-Yevick"),
            ClosureKind::PartialSeriesExpansion(x) => {
                write!(f, "Partial Series Expansion ({} terms)", x)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum IntegralEquationKind {
    XRISM,
    DRISM,
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
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RISMDriver {
    pub vv: DataRs,
    pub uv: Option<DataRs>,
    pub operator: OperatorConfig,
    pub potential: PotentialConfig,
    pub solver: SolverConfig,
}

#[pymethods]
impl RISMDriver {
    #[new]
    fn new<'py>(
        data_config: &PyAny,
        operator_config: &PyAny,
        potential_config: &PyAny,
        solver_config: &PyAny,
    ) -> PyResult<Self> {
        // Extract problem data
        let data_config: DataConfig = data_config.extract()?;
        let (vv, uv);

        // Construct the solvent-solvent problem
        vv = DataRs::new(
            data_config.temp,
            data_config.kt,
            data_config.amph,
            data_config.nsv,
            data_config.nsv,
            data_config.nspv,
            data_config.nspv,
            data_config.npts,
            data_config.radius,
            data_config.nlambda,
            data_config.atoms.clone(),
            data_config.solvent_species.clone(),
        );

        // Check if a solute-solvent problem exists
        match data_config.nsu {
            None => uv = None,
            _ => {
                // Construct the solute-solvent problem
                uv = Some(DataRs::new(
                    data_config.temp,
                    data_config.kt,
                    data_config.amph,
                    data_config.nsu.unwrap(),
                    data_config.nsv,
                    data_config.nspu.unwrap(),
                    data_config.nspv,
                    data_config.npts,
                    data_config.radius,
                    data_config.nlambda,
                    data_config.atoms,
                    data_config.solvent_species,
                ));
            }
        }

        // Extract operator information
        let operator: OperatorConfig = operator_config.extract()?;

        // Extract potential information
        let potential: PotentialConfig = potential_config.extract()?;

        // Extract solver information
        let solver: SolverConfig = solver_config.extract()?;

        Ok(RISMDriver {
            vv,
            uv,
            operator,
            potential,
            solver,
        })
    }

    pub fn print_info(&self) {
        println!("
             ____  ___ ____  __  __ 
 _ __  _   _|  _ \\|_ _/ ___||  \\/  |
| '_ \\| | | | |_) || |\\___ \\| |\\/| |
| |_) | |_| |  _ < | | ___) | |  | |
| .__/ \\__, |_| \\_\\___|____/|_|  |_|
|_|    |___/                        

");
        match &self.uv {
            None => println!("Solvent-Solvent Problem:\n{}\n\nJob Configuration:\n{}\n{}\n{}", self.vv, self.operator, self.potential, self.solver),
            Some(uv) => println!("Solvent-Solvent Problem:\n{}\n\nSolute-Solvent Problem:\n{}\n\nJob Configuration:\n{}\n{}\n{}", self.vv, uv, self.operator, self.potential, self.solver),
        }
    }

    pub fn do_rism(&mut self) {
        todo!()
    }

    // pub fn extract<'py>(
    //     &'py self,
    //     py: Python<'py>,
    // ) -> PyResult<(
    //     &PyArray3<f64>,
    //     &PyArray3<f64>,
    //     &PyArray3<f64>,
    //     &PyArray3<f64>,
    // )> {
    //     Ok((
    //         self.data.cr.clone().into_pyarray(py),
    //         self.data.tr.clone().into_pyarray(py),
    //         self.data.hr.clone().into_pyarray(py),
    //         self.data.hk.clone().into_pyarray(py),
    //     ))
    // }
}
