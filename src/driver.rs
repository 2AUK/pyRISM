use crate::data::{
    Correlations, DataConfig, DataRs, DielectricData, Grid, Interactions, SingleData, Site,
    Species, SystemState,
};
use crate::integralequation::IntegralEquationKind;
use crate::operator::{Operator, OperatorConfig};
use crate::potential::{Potential, PotentialConfig};
use crate::solver::Solver;
use crate::solver::SolverConfig;
use log::{error, info, warn};
use ndarray::{Array, Array1, Array2, Array3, Axis, Zip};
use pyo3::prelude::*;

pub enum Verbosity {
    Quiet,
    Warning,
    Info,
    Debug,
    Trace,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RISMDriver {
    pub solvent: SingleData,
    pub solute: Option<SingleData>,
    pub data: DataConfig,
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
        let data: DataConfig = data_config.extract()?;
        let (solvent, solute);
        let shape = (data.npts, data.nsv, data.nsv);

        // Construct the solvent-solvent problem
        solvent = SingleData::new(
            data.solvent_atoms.clone(),
            data.solvent_species.clone(),
            shape,
        );

        // Check if a solute-solvent problem exists
        match data.nsu {
            None => solute = None,
            _ => {
                let shape = (data.npts, data.nsu.unwrap(), data.nsu.unwrap());
                // Construct the solute-solvent problem
                solute = Some(SingleData::new(
                    data.solute_atoms.as_ref().unwrap().clone(),
                    data.solute_species.as_ref().unwrap().clone(),
                    shape,
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
            solvent,
            solute,
            data,
            operator,
            potential,
            solver,
        })
    }

    pub fn execute(&mut self) {
        self.print_header();
        simple_logger::init_with_env().unwrap();
        // set up operator(RISM equation and Closure)
        info!("Defining operator");
        let operator = Operator::new(&self.operator);

        let (mut vv, uv) = self.problem_setup();

        let mut solver = self.solver.solver.set(&self.solver.settings);

        match solver.solve(&mut vv, &operator) {
            Ok(()) => info!("Finished!"),
            Err(e) => error!("{}", e),
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
    fn problem_setup(&mut self) -> (DataRs, Option<DataRs>) {
        let (mut vv_problem, uv_problem);
        info!("Defining solvent-solvent problem");
        let grid = Grid::new(self.data.npts, self.data.radius);
        let system = SystemState::new(
            self.data.temp,
            self.data.kt,
            self.data.amph,
            self.data.nlambda,
        );
        let dielectric;
        match self.operator.integral_equation {
            IntegralEquationKind::DRISM => {
                let mut k_exp_term = grid.kgrid.clone();
                let total_density = self.solvent.species.iter().fold(0.0,|acc, species| acc + species.dens); 
                let drism_damping = self.data.drism_damping.expect("damping parameter for DRISM set");
                let diel = self.data.dielec.expect("dielectric constant set");
                k_exp_term.par_mapv_inplace(|x| (-1.0 * (drism_damping * x / 2.0).powf(2.0)).exp());
                let y = 0.0;
                let hc0 = (((diel - 1.0) / y) - 3.0) / total_density;
                let hck = hc0 * k_exp_term;

                let chi = {
                    let mut d0x = Array::zeros(self.data.nsv);
                    let mut d0y = Array::zeros(self.data.nsv);
                    let mut d1z = Array::zeros(self.data.nsv);
                    let mut chi = Array::zeros((self.data.npts, self.data.nsv, self.data.nsv));
                    let mut i = 0;
                    for (ki, k) in grid.kgrid.iter().enumerate() {
                        for species in self.solvent.species.iter() {
                            for atm in species.atom_sites.iter() {
                                
                                let k_coord = *k * Array::from_vec(atm.coords.clone());

                                if k_coord[0] == 0.0 {
                                    d0x[i] = 1.0
                                } else {
                                    d0x[i] = (k_coord[0]).sin() / k_coord[0];
                                }

                                if k_coord[1] == 0.0 {
                                    d0y[i] = 1.0
                                } else {
                                    d0y[i] = (k_coord[1]).sin() / k_coord[1];
                                }

                                if k_coord[2] == 0.0 {
                                    d1z[i] = 1.0
                                } else {
                                    d1z[i] = ((k_coord[2].sin() / k_coord[2]) - k_coord[2].cos()) / k_coord[2];
                                }

                                i += 1;
                            }
                        }

                        for i in 0..self.data.nsv {
                            for j in 0..self.data.nsv {
                                chi[[ki, i, j]] = d0x[i] * d0y[i] * d1z[i] * d0x[j] * d0y[j] * d1z[j] * hck[ki];
                            }
                        }
                    }

                    

                };

                

                dielectric = Some(DielectricData::new(
                    drism_damping,
                    diel,
                    (self.data.npts, self.data.nsv, self.data.nsv),
                ));
            }
            _ => dielectric = None,
        }
        
        let interactions = Interactions::new(self.data.npts, self.data.nsv, self.data.nsv);
        let correlations = Correlations::new(self.data.npts, self.data.nsv, self.data.nsv);

        vv_problem = DataRs::new(
            system.clone(),
            self.solvent.clone(),
            self.solvent.clone(),
            grid.clone(),
            interactions,
            correlations,
            dielectric.clone(),
        );

        info!("Tabulating solvent-solvent potentials");
        self.build_potential(&mut vv_problem);

        info!("Tabulating solvent intramolecular correlation functions");
        self.build_intramolecular_correlation(&mut vv_problem);
        match &self.solute {
            None => {
                info!("No solute data");
                uv_problem = None
            }
            Some(solute) => {
                info!("Solute data found");
                warn!("pyRISM only tested for infinite dilution - setting non-zero solute density may not work");
                let interactions =
                    Interactions::new(self.data.npts, self.data.nsu.unwrap(), self.data.nsv);
                let correlations =
                    Correlations::new(self.data.npts, self.data.nsu.unwrap(), self.data.nsv);
                let mut uv = DataRs::new(
                    system.clone(),
                    solute.clone(),
                    self.solvent.clone(),
                    grid.clone(),
                    interactions,
                    correlations,
                    dielectric.clone(),
                );
                info!("Tabulating solute-solvent potentials");
                self.build_potential(&mut uv);

                info!("Tabulating solute intramolecular correlation functions");
                self.build_intramolecular_correlation(&mut uv);

                uv_problem = Some(uv)
            }
        }
        (vv_problem, uv_problem)
    }

    fn print_header(&self) {
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
    }

    fn build_potential(&mut self, problem: &mut DataRs) {
        let potential = Potential::new(&self.potential);
        // set up total potential
        let npts = problem.grid.npts;
        let num_sites_a = problem.data_a.sites.len();
        let num_sites_b = problem.data_b.sites.len();
        let shape = (npts, num_sites_a, num_sites_b);
        let (mut u_nb, mut u_c) = (Array::zeros(shape), Array::zeros(shape));
        // compute nonbonded interaction
        (potential.nonbonded)(
            &problem.data_a.sites,
            &problem.data_b.sites,
            &problem.grid.rgrid,
            &mut u_nb,
        );
        // compute electrostatic interaction
        (potential.coulombic)(
            &problem.data_a.sites,
            &problem.data_b.sites,
            &problem.grid.rgrid,
            &mut u_c,
        );
        u_c *= problem.system.amph;
        // set total interaction
        problem.interactions.ur = u_nb + u_c;

        // compute renormalised potentials
        (potential.renormalisation_real)(
            &problem.data_a.sites,
            &problem.data_b.sites,
            &problem.grid.rgrid,
            &mut problem.interactions.ur_lr,
        );
        (potential.renormalisation_fourier)(
            &problem.data_a.sites,
            &problem.data_b.sites,
            &problem.grid.kgrid,
            &mut problem.interactions.uk_lr,
        );

        problem.interactions.ur_lr *= problem.system.amph;
        problem.interactions.uk_lr *= problem.system.amph;

        // set short range interactions
        problem.interactions.u_sr = &problem.interactions.ur - &problem.interactions.ur_lr;
    }

    fn build_intramolecular_correlation(&mut self, problem: &mut DataRs) {
        let distances_a = Self::distance_matrix(&problem.data_a.species, &problem.data_a.species);
        let distances_b = Self::distance_matrix(&problem.data_b.species, &problem.data_b.species);

        Self::intramolecular_corr_impl(&distances_a, &problem.grid.kgrid, &mut problem.data_a.wk);
        Self::intramolecular_corr_impl(&distances_b, &problem.grid.kgrid, &mut problem.data_b.wk);
    }

    fn intramolecular_corr_impl(
        distances: &Array2<f64>,
        k: &Array1<f64>,
        out_array: &mut Array3<f64>,
    ) {
        let (npts, num_sites_a, num_sites_b) = out_array.dim();
        let one = Array::ones(npts);
        let zero = Array::zeros(npts);

        Zip::from(out_array.lanes_mut(Axis(0)))
            .and(distances)
            .par_for_each(|mut lane, elem| {
                let elem = *elem;
                if elem < 0.0 {
                    lane.assign(&zero)
                } else if elem == 0.0 {
                    lane.assign(&one)
                } else {
                    let arr = (elem * k).mapv(|a| a.sin()) / (elem * k);
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
