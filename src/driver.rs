use crate::data::{
    Correlations, DataConfig, DataRs, DielectricData, Grid, Interactions, SingleData, Species,
    SystemState,
};
use crate::dipole::*;
use crate::input::{Configuration, InputTOMLHandler};
use crate::integralequation::IntegralEquationKind;
use crate::operator::{Operator, OperatorConfig};
use crate::potential::{Potential, PotentialConfig};
use crate::solution::*;
use crate::solver::SolverConfig;
use bzip2::read::{BzDecoder, BzEncoder};
use bzip2::Compression;
use gnuplot::{AxesCommon, Caption, Color, Figure, Fix, LineWidth};
use log::{debug, error, info, warn};
use ndarray::{s, Array, Array1, Array2, Array3, Axis, Slice, Zip};
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::fs;
use std::io::prelude::*;
use std::path::PathBuf;

pub enum Verbosity {
    Quiet,
    Verbose,
    VeryVerbose,
}

// fn plot(x: &Array1<f64>, y: &Array1<f64>) {
//     let x_vec = x.to_vec();
//     let y_vec = y.to_vec();
//     let mut fg = Figure::new();
//     fg.axes2d()
//         .lines(&x_vec, &y_vec, &[LineWidth(1.5), Color("red")]);
//     fg.show().unwrap();
// }
//
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

    pub fn do_rism<'py>(&'py mut self, py: Python<'py>) {
        // -> PyResult<Py<PyAny>> {
        self.execute();
        // Ok(PyCorrelations::new(
        //     uv.clone().unwrap().correlations.cr,
        //     uv.clone().unwrap().correlations.tr,
        //     uv.clone().unwrap().correlations.hr,
        //     gr_uv,
        //     py,
        // )
        // .into_py(py))
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
    pub fn execute(&mut self) {
        self.print_header();
        simple_logger::init_with_env().unwrap();
        // set up operator(RISM equation and Closure)
        info!("Defining operator");
        let operator = Operator::new(&self.operator);
        let operator_uv = Operator::new(&OperatorConfig {
            integral_equation: IntegralEquationKind::UV,
            ..self.operator.clone()
        });

        println!(
            "{:#?}\n\n{:#?}\n\n{:#?}\n\n{:#?}",
            self.data, self.operator, self.solver, self.potential
        );

        let (mut vv, mut uv) = self.problem_setup();

        let mut solver = self.solver.solver.set(&self.solver.settings);

        match solver.solve(&mut vv, &operator) {
            Ok(s) => info!("{}", s),
            Err(e) => error!("{}", e),
        }

        let vv_solution = SolvedData::new(
            self.data.clone(),
            self.solver.clone(),
            self.potential.clone(),
            self.operator.clone(),
            vv.interactions.clone(),
            vv.correlations.clone(),
        );

        let gr = &vv.correlations.cr + &vv.correlations.tr + 1.0;

        match uv {
            None => info!("No solute-solvent problem"),
            Some(ref mut uv) => {
                uv.solution = Some(vv_solution.clone());
                match solver.solve(uv, &operator_uv) {
                    Ok(s) => info!("{}", s),
                    Err(e) => panic!("{}", e),
                }
            }
        }

        let uv_solution = SolvedData::new(
            self.data.clone(),
            self.solver.clone(),
            self.potential.clone(),
            self.operator.clone(),
            uv.clone().unwrap().interactions,
            uv.clone().unwrap().correlations,
        );

        let gr_uv =
            &uv.clone().unwrap().correlations.cr + &uv.clone().unwrap().correlations.tr + 1.0;

        let encoded_vv: Vec<u8> =
            bincode::serialize(&vv_solution).expect("encode solvent-solvent results to binary");
        let compressor = BzEncoder::new(encoded_vv.as_slice(), Compression::best());
    }

    pub fn from_toml(fname: PathBuf) -> Self {
        let config: Configuration = InputTOMLHandler::construct_configuration(&fname);
        let data = config.data_config;
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
        let operator: OperatorConfig = config.operator_config;

        // Extract potential information
        let potential: PotentialConfig = config.potential_config;

        // Extract solver information
        let solver: SolverConfig = config.solver_config;

        RISMDriver {
            solvent,
            solute,
            data,
            operator,
            potential,
            solver,
        }
    }
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
        info!("Checking for dipole moment");
        let mut dm_vec = Vec::new();
        for species in self.solvent.species.iter() {
            dm_vec.push(dipole_moment(&species.atom_sites));
        }
        let (dm, _): (Vec<_>, Vec<_>) = dm_vec.into_iter().partition(Result::is_ok);
        if dm.is_empty() {
            warn!("No dipole moment found!")
        } else if dm.is_empty() && self.operator.integral_equation == IntegralEquationKind::DRISM {
            warn!("No dipole moment found! Switch from DRISM to XRISM");
            self.operator.integral_equation = IntegralEquationKind::XRISM;
        } else {
            info!("Aligning dipole moment to z-axis");
            for species in self.solvent.species.iter_mut() {
                let tot_charge = total_charge(&species.atom_sites);
                let mut coc = centre_of_charge(&species.atom_sites);
                coc /= tot_charge;
                translate(&mut species.atom_sites, &coc);
                match reorient(&mut species.atom_sites) {
                    Ok(_) => (),
                    Err(e) => panic!(
                        "{}; there should be a dipole moment present for this step, something has gone FATALLY wrong",
                        e
                    ),
                }
            }
        }
        let dielectric;
        match self.operator.integral_equation {
            IntegralEquationKind::DRISM => {
                info!("Calculating dielectric asymptotics for DRISM");
                let mut k_exp_term = grid.kgrid.clone();
                let total_density = self
                    .solvent
                    .species
                    .iter()
                    .fold(0.0, |acc, species| acc + species.dens);
                let drism_damping = self
                    .data
                    .drism_damping
                    .expect("damping parameter for DRISM set");
                let diel = self.data.dielec.expect("dielectric constant set");
                k_exp_term.par_mapv_inplace(|x| (-1.0 * (drism_damping * x / 2.0).powf(2.0)).exp());
                let dipole_density = self.solvent.species.iter().fold(0.0, |acc, species| {
                    match dipole_moment(&species.atom_sites) {
                        Ok((_, dm)) => acc + species.dens * dm * dm,
                        _ => acc + 0.0,
                    }
                });
                let y = 4.0 * PI * dipole_density / 9.0;
                let hc0 = (((diel - 1.0) / y) - 3.0) / total_density;
                let hck = hc0 * k_exp_term;
                debug!("y: {}", y);
                debug!("h_c(0): {}", hc0);

                let chi = {
                    let mut d0x = Array::zeros(self.data.nsv);
                    let mut d0y = Array::zeros(self.data.nsv);
                    let mut d1z = Array::zeros(self.data.nsv);
                    let mut chi = Array::zeros((self.data.npts, self.data.nsv, self.data.nsv));
                    for (ki, k) in grid.kgrid.iter().enumerate() {
                        let mut i = 0;
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
                                    d1z[i] = ((k_coord[2].sin() / k_coord[2]) - k_coord[2].cos())
                                        / k_coord[2];
                                }

                                i += 1;
                            }
                        }
                        for i in 0..self.data.nsv {
                            for j in 0..self.data.nsv {
                                chi[[ki, i, j]] =
                                    d0x[i] * d0y[i] * d1z[i] * d0x[j] * d0y[j] * d1z[j] * hck[ki];
                            }
                        }
                    }
                    chi
                };
                dielectric = Some(DielectricData::new(drism_damping, diel, chi));
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
                warn!(
                    "pyRISM only tested for infinite dilution; setting non-zero solute density may not work"
                );
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
        let (npts, _, _) = out_array.dim();
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
