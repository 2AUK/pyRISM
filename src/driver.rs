use crate::data::{DataConfig, DataRs, Site};
use crate::operator::{Operator, OperatorConfig};
use crate::potential::{Potential, PotentialConfig};
use crate::solver::SolverConfig;
use ndarray::{Array, Array3};
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

    pub fn execute(&mut self) {
        // set up operator(RISM equation and Closure)
        let operator = Operator::new(&self.operator);

        // build potentials
        self.build_vv_potential();



        self.print_info()

    }

    fn do_rism(&mut self) {
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

impl RISMDriver {
    fn print_info(&self) {
        println!(
            "
             ____  ___ ____  __  __ 
 _ __  _   _|  _ \\|_ _/ ___||  \\/  |
| '_ \\| | | | |_) || |\\___ \\| |\\/| |
| |_) | |_| |  _ < | | ___) | |  | |
| .__/ \\__, |_| \\_\\___|____/|_|  |_|
|_|    |___/                        

"
        );
        match &self.uv {
            None => println!("Solvent-Solvent Problem:\n{}\n\nJob Configuration:\n{}\n{}\n{}", self.vv, self.operator, self.potential, self.solver),
            Some(uv) => println!("Solvent-Solvent Problem:\n{}\n\nSolute-Solvent Problem:\n{}\n\nJob Configuration:\n{}\n{}\n{}", self.vv, uv, self.operator, self.potential, self.solver),
        }
    }

    fn build_vv_potential(&mut self) {
        let potential = Potential::new(&self.potential);
        // set up total potential
        let (mut u_nb, mut u_c) = (Array::zeros(self.vv.ur.raw_dim()), Array::zeros(self.vv.ur.raw_dim()));
        // compute nonbonded interaction
        (potential.nonbonded)(&self.vv.sites, &self.vv.sites, &self.vv.grid.rgrid, &mut u_nb);
        // compute electrostatic interaction
        (potential.nonbonded)(&self.vv.sites, &self.vv.sites, &self.vv.grid.rgrid, &mut u_c);
        // set total interaction
        self.vv.ur = u_nb + u_c;

        // compute renormalised potentials
        (potential.renormalisation_real)(&self.vv.sites, &self.vv.sites, &self.vv.grid.rgrid, &mut self.vv.ur_lr);
        (potential.renormalisation_real)(&self.vv.sites, &self.vv.sites, &self.vv.grid.kgrid, &mut self.vv.uk_lr);
    }

    fn build_uv_potential(&mut self) {
        let potential = Potential::new(&self.potential);
        let uv = self.uv.as_mut().unwrap();
        // set up total potential
        let (mut u_nb, mut u_c) = (Array::zeros(uv.ur.raw_dim()), Array::zeros(uv.ur.raw_dim()));
        // compute nonbonded interaction
        (potential.nonbonded)(&uv.sites, &self.vv.sites, &uv.grid.rgrid, &mut u_nb);
        // compute electrostatic interaction
        (potential.nonbonded)(&uv.sites, &self.vv.sites, &uv.grid.rgrid, &mut u_c);
        // set total interaction
        uv.ur = u_nb + u_c;

        // compute renormalised potentials
        (potential.renormalisation_real)(&uv.sites, &self.vv.sites, &self.vv.grid.rgrid, &mut uv.ur_lr);
        (potential.renormalisation_real)(&uv.sites, &self.vv.sites, &self.vv.grid.kgrid, &mut uv.uk_lr);
    }
}
