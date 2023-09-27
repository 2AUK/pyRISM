use crate::data::{DataConfig, DataRs, Site, Species};
use crate::operator::{Operator, OperatorConfig};
use crate::potential::{Potential, PotentialConfig};
use crate::solver::SolverConfig;
use ndarray::{Array, Array1, Array2, Array3, Axis, Zip};
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
            data_config.solvent_atoms.clone(),
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
                    data_config.solute_atoms.unwrap().clone(),
                    data_config.solute_species.unwrap().clone(),
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
        self.print_info();
        // set up operator(RISM equation and Closure)
        println!("\nDefining operator...");
        let operator = Operator::new(&self.operator);

        // build potentials
        self.build_vv_potential();

        // compute intramolecular correlation function
        self.build_vv_intramolecular_correlation();

        match self.uv {
            None => println!("\nNo solute-solvent data..."),
            _ => {
                self.build_uv_potential();
                self.build_uu_intramolecular_correlation();
            }
        }
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
        println!("\nBuilding solvent-solvent potentials...");
        let potential = Potential::new(&self.potential);
        // set up total potential
        let (mut u_nb, mut u_c) = (
            Array::zeros(self.vv.ur.raw_dim()),
            Array::zeros(self.vv.ur.raw_dim()),
        );
        // compute nonbonded interaction
        println!("\t{}...", self.potential.nonbonded);
        (potential.nonbonded)(
            &self.vv.sites,
            &self.vv.sites,
            &self.vv.grid.rgrid,
            &mut u_nb,
        );
        // compute electrostatic interaction
        println!("\t{}...", self.potential.coulombic);
        (potential.coulombic)(
            &self.vv.sites,
            &self.vv.sites,
            &self.vv.grid.rgrid,
            &mut u_c,
        );
        // set total interaction
        self.vv.ur = u_nb + self.vv.amph * u_c;

        // compute renormalised potentials
        println!("\t{}...", self.potential.renormalisation_real);
        (potential.renormalisation_real)(
            &self.vv.sites,
            &self.vv.sites,
            &self.vv.grid.rgrid,
            &mut self.vv.ur_lr,
        );
        println!("\t{}...", self.potential.renormalisation_fourier);
        (potential.renormalisation_real)(
            &self.vv.sites,
            &self.vv.sites,
            &self.vv.grid.kgrid,
            &mut self.vv.uk_lr,
        );
    }

    fn build_uv_potential(&mut self) {
        println!("\nBuilding solute-solvent potentials...");
        let potential = Potential::new(&self.potential);
        let uv = self.uv.as_mut().unwrap();
        // set up total potential
        let (mut u_nb, mut u_c) = (Array::zeros(uv.ur.raw_dim()), Array::zeros(uv.ur.raw_dim()));
        // compute nonbonded interaction
        println!("\t{}...", self.potential.nonbonded);
        (potential.nonbonded)(&uv.sites, &self.vv.sites, &uv.grid.rgrid, &mut u_nb);
        // compute electrostatic interaction
        println!("\t{}...", self.potential.coulombic);
        (potential.nonbonded)(&uv.sites, &self.vv.sites, &uv.grid.rgrid, &mut u_c);
        // set total interaction
        uv.ur = u_nb + uv.amph * u_c;

        // compute renormalised potentials
        println!("\t{}...", self.potential.renormalisation_real);
        (potential.renormalisation_real)(
            &uv.sites,
            &self.vv.sites,
            &self.vv.grid.rgrid,
            &mut uv.ur_lr,
        );
        println!("\t{}...", self.potential.renormalisation_fourier);
        (potential.renormalisation_real)(
            &uv.sites,
            &self.vv.sites,
            &self.vv.grid.kgrid,
            &mut uv.uk_lr,
        );
    }

    fn build_vv_intramolecular_correlation(&mut self) {
        println!("\nBuilding solvent-solvent intramolecular correlation matrix...");
        let distances = Self::distance_matrix(&self.vv.species, &self.vv.species);
        let one = Array::ones(self.vv.grid.npts);
        let zero = Array::zeros(self.vv.grid.npts);

        Zip::from(self.vv.wk.lanes_mut(Axis(0)))
            .and(&distances)
            .par_for_each(|mut lane, elem| {
                let elem = *elem;
                if elem < 0.0 {
                    lane.assign(&zero)
                } else if elem == 0.0 {
                    lane.assign(&one)
                } else {
                    let arr = (elem * &self.vv.grid.kgrid).mapv(|a| a.sin())
                        / (elem * &self.vv.grid.kgrid);
                    lane.assign(&arr)
                }
            });
    }

    fn build_uu_intramolecular_correlation(&mut self) {
        println!("\nBuilding solute-solute intramolecular correlation matrix...");
        let uv = self.uv.as_mut().unwrap();
        let distances = Self::distance_matrix(&uv.species.clone(), &uv.species.clone());
        let one = Array::ones(uv.grid.npts);
        let zero = Array::zeros(uv.grid.npts);

        Zip::from(uv.wk.lanes_mut(Axis(0)))
            .and(&distances)
            .par_for_each(|mut lane, elem| {
                let elem = *elem;
                if elem < 0.0 {
                    lane.assign(&zero)
                } else if elem == 0.0 {
                    lane.assign(&one)
                } else {
                    let arr = (elem * &uv.grid.kgrid).mapv(|a| a.sin()) / (elem * &uv.grid.kgrid);
                    lane.assign(&arr)
                }
            });
    }

    fn distance(a: Array1<f64>, b: Array1<f64>) -> f64 {
        (a - b).mapv(|a| a.powf(2.0)).sum().sqrt()
    }

    fn distance_matrix(species_a: &[Species], species_b: &[Species]) -> Array2<f64> {
        let ns1 = species_a.iter().fold(0, |acc, spec| acc + spec.ns);
        let ns2 = species_b.iter().fold(0, |acc, spec| acc + spec.ns);
        let mut out_mat = Array::zeros((ns1, ns2));
        let mut i = 0;
        for isp in species_a.iter() {
            for iat in isp.atom_sites.iter() {
                let mut j = 0;
                for jsp in species_b.iter() {
                    for jat in jsp.atom_sites.iter() {
                        if isp != jsp {
                            out_mat[[i, j]] = -1.0;
                        } else {
                            let dist = Self::distance(
                                Array::from_shape_vec(3, iat.coords.clone()).unwrap(),
                                Array::from_shape_vec(3, jat.coords.clone()).unwrap(),
                            );
                            out_mat[[i, j]] = dist;
                        }
                        j += 1;
                    }
                }
                i += 1;
            }
        }
        out_mat
    }
}
