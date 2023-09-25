use crate::data::{DataConfig, DataRs};
use crate::solver::SolverConfig;
use crate::potential::PotentialConfig;
use crate::operator::OperatorConfig;
use pyo3::prelude::*;

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
