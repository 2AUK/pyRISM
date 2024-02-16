use crate::data::solution::*;
use crate::data::{
    configuration::{
        Configuration,
        {operator::OperatorConfig, potential::PotentialConfig, problem::ProblemConfig, solver::*},
    },
    core::*,
};
use crate::grids::radial_grid::Grid;
use crate::iet::integralequation::IntegralEquationKind;
use crate::iet::operator::Operator;
use crate::interactions::dipole::*;
use crate::interactions::potential::Potential;
use crate::io::input::InputTOMLHandler;
use crate::structure::system::Species;
use flate2::{read, write, Compression};
use log::{info, trace, warn};
use ndarray::{Array, Array1, Array2, Array3, Axis, Zip};
use pyo3::prelude::*;
use std::cell::RefCell;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;
use tabled::builder::Builder;

// Feature for switching on allocation profiler
#[cfg(feature = "dhat-on")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Struct for storing job diagnostics
pub struct JobDiagnostics {
    pub vv_time: f64,
    pub uv_time: f64,
    pub vv_iterations: usize,
    pub uv_iterations: usize,
    pub job_time: f64,
}

impl std::fmt::Display for JobDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Solvent-Solvent Solver Time: {} s\nSolvent-Solvent Solver Iterations: {}\nSolute-Solvent Solver Time: {} s\nSolute-Solvent Solver Iterations: {}\nTotal Job Time: {} s",
            self.vv_time, self.vv_iterations, self.uv_time, self.uv_iterations, self.job_time
        )
    }
}

/// Verbosity flags for stdout
pub enum Verbosity {
    /// No output
    Quiet,
    /// INFO logging only
    Verbose,
    /// TRACE logging and job details
    VeryVerbose,
}

/// Compression flag for compressing solvent-solvent problem
pub enum Compress {
    /// Compress the problem to `.bin`. Compressed file takes input file name.
    Compress,
    /// Do not compress. Driver will switch to this if already using a compressed `.bin` file.
    NoCompress,
}

/// The RISMDriver struct solves the solvent-solvent and/or solute-solvent problem. It constructs
/// all required input functions from the details it reads from the input file passed to it. It
/// then runs the chosen solver and returns the solved problem.
///
/// Solvent and solute information is stored and the details of the full problem, the operator (i.e. the
/// integral equation and closure), the potential energy function and the solver are passed as
/// Config structs. These are used to fully construct each component of the Driver.
#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct RISMDriver {
    /// Input job name
    pub name: String,
    /// Solvent data
    pub solvent: Rc<RefCell<SingleData>>,
    /// Solute data
    pub solute: Option<Rc<RefCell<SingleData>>>,
    /// Config to set up problem
    pub data: ProblemConfig,
    /// Config to define operator
    pub operator: OperatorConfig,
    /// Config to define potential
    pub potential: PotentialConfig,
    /// Config to define solver
    pub solver: SolverConfig,
    _preconv: Option<SolvedData>,
}

impl RISMDriver {
    // fn new(name: String, config: Configuration) -> Self {
    //     todo!()
    // }

    /// Restore the renormalised functions by adding back the long-range component to $c(r)$ and
    /// $t(r)$
    fn restore_renormalised_functions(data: &mut SolvedData, beta: f64) {
        let cr = data.correlations.cr.clone();
        let tr = data.correlations.tr.clone();
        let urlr = data.interactions.ur_lr.clone();

        data.correlations.cr.assign(&(&cr - beta * &urlr));
        data.correlations.tr.assign(&(&tr + beta * &urlr));
        data.correlations.hr = &data.correlations.cr + &data.correlations.tr;
    }

    /// Sets the verbosity level, initialises problem based on Config structs and returns
    /// solutions.
    pub fn execute(
        &mut self,
        verbosity: &Verbosity,
        compress: &Compress,
    ) -> (Solutions, JobDiagnostics) {
        match verbosity {
            Verbosity::Quiet => (),
            Verbosity::Verbose => {
                self.print_header();
                simple_logger::SimpleLogger::new()
                    .with_level(log::Level::Info.to_level_filter())
                    .with_timestamp_format(time::macros::format_description!(
                        "[hour]:[minute]:[second]"
                    ))
                    .init()
                    .unwrap();
            }
            Verbosity::VeryVerbose => {
                self.print_header();
                self.print_job_details();
                simple_logger::SimpleLogger::new()
                    .with_level(log::Level::Trace.to_level_filter())
                    .with_timestamp_format(time::macros::format_description!(
                        "[hour]:[minute]:[second]"
                    ))
                    .init()
                    .unwrap();
            }
        }
        let timer = Instant::now();

        let (mut vv, mut uv) = self.problem_setup();

        // set up operator(RISM equation and Closure)
        trace!("Defining operator");
        let operator = Operator::new(&self.operator);
        let operator_uv = Operator::new(&OperatorConfig {
            integral_equation: IntegralEquationKind::UV,
            ..self.operator.clone()
        });

        let mut solver = self.solver.solver.set(&self.solver.settings);

        let mut vv_iterations = 0;
        let mut uv_iterations = 0;
        let mut vv_time = 0.0;
        let mut uv_time = 0.0;
        let mut vv_solution = match self._preconv {
            None => {
                vv_iterations = 0;
                info!("Starting solvent-solvent solver");
                let timer = Instant::now();
                for ilam in 1..self.data.nlambda + 1 {
                    let lam = 1.0 * ilam as f64 / self.data.nlambda as f64;
                    vv.system.curr_lam = lam;
                    info!("Lambda cycle: {}/{}", ilam, self.data.nlambda);
                    match solver.solve(&mut vv, &operator) {
                        Ok(s) => {
                            info!("{}", s);
                            vv_iterations += s.0
                        }
                        Err(e) => panic!("{}", e),
                    };
                }

                vv_time = timer.elapsed().as_secs_f64();

                info!(
                    "Total Solvent-Solvent Iterations: {} Total Solver Time: {} s",
                    vv_iterations, vv_time
                );

                let vv_solution = SolvedData::new(
                    self.data.clone(),
                    self.solver.clone(),
                    self.potential.clone(),
                    self.operator.clone(),
                    vv.interactions.clone(),
                    vv.correlations.clone(),
                );
                if let Compress::Compress = compress {
                    info!("Compressing solvent-solvent data");
                    let inteq =
                        str::replace(&self.operator.integral_equation.to_string(), " ", "_");
                    let clos = str::replace(&self.operator.closure.to_string(), " ", "_");
                    let temp = &self.data.temp;
                    let name = format!("{}_{}_{}_{}K.bin", self.name, inteq, clos, temp);
                    let file = fs::File::create(name).unwrap();
                    let mut compressor = write::GzEncoder::new(file, Compression::best());
                    bincode::serialize_into(&mut compressor, &vv_solution)
                        .expect("encode solvent-solvent results to compressed binary");
                    compressor
                        .finish()
                        .expect("finish encoding solvent-solvent results");
                }
                vv_solution
            }
            Some(ref x) => {
                if let Compress::Compress = compress {
                    warn!(
                        "Already loading saved solution! Skipping solvent-solvent compression..."
                    );
                }
                x.clone()
            }
        };

        let uv_solution = match uv {
            None => {
                info!("No solute-solvent problem");
                None
            }
            Some(ref mut uv) => {
                uv_iterations = 0;
                let timer = Instant::now();
                info!("Starting solute-solvent solver");
                uv.solution = Some(vv_solution.clone());
                for ilam in 1..self.data.nlambda + 1 {
                    let lam = 1.0 * ilam as f64 / self.data.nlambda as f64;
                    uv.system.curr_lam = lam;
                    info!("Lambda cycle: {}/{}", ilam, self.data.nlambda);

                    match solver.solve(uv, &operator_uv) {
                        Ok(s) => {
                            info!("{}", s);
                            uv_iterations += s.0;
                        }
                        Err(e) => panic!("{}", e),
                    }
                }
                uv_time = timer.elapsed().as_secs_f64();
                info!(
                    "Total Solute-Solvent Iterations: {} Total Solver Time: {} s",
                    uv_iterations, uv_time
                );
                let mut uv_solution = SolvedData::new(
                    self.data.clone(),
                    self.solver.clone(),
                    self.potential.clone(),
                    self.operator.clone(),
                    uv.interactions.clone(),
                    uv.correlations.clone(),
                );
                Self::restore_renormalised_functions(&mut uv_solution, uv.system.beta);
                Some(uv_solution)
            }
        };

        let job_time = timer.elapsed().as_secs_f64();

        info!(
            "Total Iterations: {} Total Job Time: {} s",
            vv_iterations + uv_iterations,
            job_time
        );

        let config = Configuration {
            data_config: self.data.clone(),
            operator_config: self.operator.clone(),
            potential_config: self.potential.clone(),
            solver_config: self.solver.clone(),
        };

        Self::restore_renormalised_functions(&mut vv_solution, vv.system.beta);

        (
            Solutions {
                config,
                vv: vv_solution,
                uv: uv_solution,
            },
            JobDiagnostics {
                vv_time,
                uv_time,
                vv_iterations,
                uv_iterations,
                job_time,
            },
        )
    }

    fn load_preconv_data(path: &Option<PathBuf>) -> Option<SolvedData> {
        match path {
            None => None,
            Some(path) => {
                let input_bin = fs::File::open(path).unwrap();
                let mut decompressor = read::GzDecoder::new(input_bin);
                let vv_solution: SolvedData = match bincode::deserialize_from(&mut decompressor) {
                    Ok(x) => x,
                    Err(e) => panic!("{}", e),
                };
                Some(vv_solution)
            }
        }
    }

    /// Read Configs from the `.toml` file
    pub fn from_toml(fname: &PathBuf) -> Self {
        let config: Configuration = InputTOMLHandler::construct_configuration(fname);
        let name = fname
            .file_stem()
            .expect("extracting name of input job script")
            .to_str()
            .expect("converting to OsStr name to str")
            .to_string();
        let data = config.data_config;
        let (solvent, solute);
        let vv_solution = Self::load_preconv_data(&data.preconverged);
        let shape = (data.npts, data.nsv, data.nsv);

        // Construct the solvent-solvent problem
        solvent = Rc::new(RefCell::new(SingleData::new(
            data.solvent_atoms.clone(),
            data.solvent_species.clone(),
            shape,
        )));

        // Check if a solute-solvent problem exists
        match data.nsu {
            None => solute = None,
            _ => {
                let shape = (data.npts, data.nsu.unwrap(), data.nsu.unwrap());
                // Construct the solute-solvent problem
                solute = Some(Rc::new(RefCell::new(SingleData::new(
                    data.solute_atoms.as_ref().unwrap().clone(),
                    data.solute_species.as_ref().unwrap().clone(),
                    shape,
                ))));
            }
        }

        // Extract operator information
        let operator: OperatorConfig = config.operator_config;

        // Extract potential information
        let potential: PotentialConfig = config.potential_config;

        // Extract solver information
        let solver: SolverConfig = config.solver_config;

        RISMDriver {
            name,
            solvent,
            solute,
            data,
            operator,
            potential,
            solver,
            _preconv: vv_solution,
        }
    }
    fn problem_setup(&mut self) -> (DataRs, Option<DataRs>) {
        let (mut vv_problem, uv_problem);
        trace!("Defining solvent-solvent problem");
        let grid = Grid::new(self.data.npts, self.data.radius);
        let system = SystemState::new(
            self.data.temp,
            self.data.kt,
            self.data.amph,
            self.data.nlambda,
        );

        let vv_problem = match self._preconv.clone() {
            None => {
                let dielectric = self.compute_dielectrics(&grid);
                vv_problem = DataRs::new(
                    system.clone(),
                    self.solvent.clone(),
                    self.solvent.clone(),
                    grid.clone(),
                    Interactions::new(self.data.npts, self.data.nsv, self.data.nsv),
                    Correlations::new(self.data.npts, self.data.nsv, self.data.nsv),
                    dielectric.clone(),
                );

                trace!("Tabulating solvent-solvent potentials");
                self.build_potential(&mut vv_problem);

                trace!("Tabulating solvent intramolecular correlation functions");
                self.build_intramolecular_correlation(&mut vv_problem);
                vv_problem
            }
            Some(ref data) => {
                trace!(
                    "Loading saved solution data from: {}",
                    std::fs::canonicalize(self.data.preconverged.clone().unwrap())
                        .expect("resolving path for binary")
                        .display()
                );
                self.data.nsv = data.data_config.nsv;
                self.data.nspv = data.data_config.nspv;
                self.solvent.borrow_mut().sites = data.data_config.solvent_atoms.clone();
                self.solvent.borrow_mut().species = data.data_config.solvent_species.clone();
                self.data.solvent_atoms = data.data_config.solvent_atoms.clone();
                self.data.solvent_species = data.data_config.solvent_species.clone();
                let new_shape = (
                    data.data_config.npts,
                    data.data_config.nsv,
                    data.data_config.nsv,
                );
                self.solvent.borrow_mut().density = {
                    let mut dens_vec: Vec<f64> = Vec::new();
                    for i in data.data_config.solvent_species.clone().into_iter() {
                        for _j in i.atom_sites {
                            dens_vec.push(i.dens);
                        }
                    }
                    Array2::from_diag(&Array::from_vec(dens_vec))
                };
                self.solvent.borrow_mut().wk = Array::zeros(new_shape);
                let dielectric = self.compute_dielectrics(&grid);
                vv_problem = DataRs::new(
                    system.clone(),
                    self.solvent.clone(),
                    self.solvent.clone(),
                    grid.clone(),
                    data.interactions.clone(),
                    data.correlations.clone(),
                    dielectric.clone(),
                );
                trace!("Rebuilding solvent intramolecular correlation function");
                self.build_intramolecular_correlation(&mut vv_problem);
                vv_problem
            }
        };
        trace!("Defining solute-solvent problem");
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
                    None,
                );
                trace!("Tabulating solute-solvent potentials");
                self.build_potential(&mut uv);

                trace!("Tabulating solute intramolecular correlation functions");
                self.build_intramolecular_correlation(&mut uv);

                uv_problem = Some(uv)
            }
        }
        (vv_problem, uv_problem)
    }

    fn compute_dielectrics(&mut self, grid: &Grid) -> Option<DielectricData> {
        trace!("Checking for dipole moment");
        let mut dm_vec = Vec::new();
        for species in self.solvent.borrow().species.iter() {
            dm_vec.push(dipole_moment(&species.atom_sites));
        }
        let (dm, _): (Vec<_>, Vec<_>) = dm_vec.into_iter().partition(Result::is_ok);
        if dm.is_empty() && self.operator.integral_equation == IntegralEquationKind::XRISM {
            warn!("No dipole moment found!")
        } else if dm.is_empty() && self.operator.integral_equation == IntegralEquationKind::DRISM {
            warn!("No dipole moment found! Switch from DRISM to XRISM");
            self.operator.integral_equation = IntegralEquationKind::XRISM;
        }
        match self.operator.integral_equation {
            IntegralEquationKind::DRISM => {
                trace!("Aligning dipole moment to z-axis");
                for species in self.solvent.borrow_mut().species.iter_mut() {
                    let tot_charge = total_charge(&species.atom_sites);
                    let mut coc = centre_of_charge(&species.atom_sites);
                    coc /= tot_charge;
                    translate(&mut species.atom_sites, &coc);
                    match reorient(&mut species.atom_sites) {
                        Ok(_) => (),
                        Err(e) => trace!("{} - {}", e, species.species_name),
                    }
                }
                trace!("Calculating dielectric asymptotics for DRISM");
                let mut k_exp_term = grid.kgrid.clone();
                let _total_density = self
                    .solvent
                    .borrow()
                    .species
                    .iter()
                    .fold(0.0, |acc, species| acc + species.dens);
                let total_site_density = {
                    let mut acc = 0.0;
                    for i in self.solvent.borrow().species.iter() {
                        for _j in i.atom_sites.iter() {
                            acc += i.dens;
                        }
                    }
                    acc
                };

                let drism_damping = self
                    .data
                    .drism_damping
                    .expect("damping parameter for DRISM set");
                let diel = self.data.dielec.expect("dielectric constant set");
                k_exp_term.par_mapv_inplace(|x| (-1.0 * (drism_damping * x / 2.0).powf(2.0)).exp());
                let _dipole_density =
                    self.solvent
                        .borrow()
                        .species
                        .iter()
                        .fold(0.0, |acc, species| {
                            match dipole_moment(&species.atom_sites) {
                                Ok((_, dm)) => acc + species.dens * dm * dm,
                                _ => acc + 0.0,
                            }
                        });
                let dipole_site_density = {
                    let mut acc = 0.0;
                    for i in self.solvent.borrow().species.iter() {
                        for _j in i.atom_sites.iter() {
                            match dipole_moment(&i.atom_sites) {
                                Ok((_, dm)) => acc += i.dens * dm * dm,
                                _ => acc += 0.0,
                            };
                        }
                    }
                    acc
                };
                let y = 4.0 * PI * dipole_site_density / 9.0;
                let hc0 = (((diel - 1.0) / y) - 3.0) / total_site_density;
                let hck = hc0 * k_exp_term;

                let chi = {
                    let mut d0x = Array::zeros(self.data.nsv);
                    let mut d0y = Array::zeros(self.data.nsv);
                    let mut d1z = Array::zeros(self.data.nsv);
                    let mut chi = Array::zeros((self.data.npts, self.data.nsv, self.data.nsv));
                    for (ki, k) in grid.kgrid.iter().enumerate() {
                        let mut i = 0;
                        for species in self.solvent.borrow().species.iter() {
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
                                    d1z[i] = 0.0
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
                Some(DielectricData::new(drism_damping, diel, chi))
            }
            _ => None,
        }
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

    fn print_job_details(&self) {
        let mut system_table_builder = Builder::new();

        system_table_builder.push_record(["System"]);
        system_table_builder.push_record(["Temperature", format!("{} K", self.data.temp).as_str()]);
        system_table_builder.push_record(["Num. Points", format!("{}", self.data.npts).as_str()]);
        system_table_builder.push_record(["Radius", format!("{} Å", self.data.radius).as_str()]);
        system_table_builder.push_record([
            "Grid Spacing",
            format!("{} Å", self.data.radius / self.data.npts as f64).as_str(),
        ]);

        system_table_builder
            .push_record(["Lambda Cycles", format!("{}", self.data.nlambda).as_str()]);

        let system_table = system_table_builder.build().to_string();

        println!("{}", system_table);

        let mut operator_table_builder = Builder::new();

        operator_table_builder.push_record(["Operator"]);
        operator_table_builder.push_record([
            "Integral Equation",
            self.operator.integral_equation.to_string().as_str(),
        ]);
        operator_table_builder.push_record(["Closure", self.operator.closure.to_string().as_str()]);

        let operator_table = operator_table_builder.build().to_string();

        println!("{}", operator_table);

        let mut potential_table_builder = Builder::new();

        potential_table_builder.push_record(["Potential"]);
        potential_table_builder
            .push_record(["Non-Bonded", self.potential.nonbonded.to_string().as_str()]);
        potential_table_builder
            .push_record(["Coulombic", self.potential.coulombic.to_string().as_str()]);

        let potential_table = potential_table_builder.build().to_string();

        println!("{}", potential_table);

        let mut solver_table_builder = Builder::new();

        solver_table_builder.push_record(["Solver"]);
        solver_table_builder.push_record(["Method", self.solver.solver.to_string().as_str()]);
        solver_table_builder.push_record([
            "Max Iterations",
            self.solver.settings.max_iter.to_string().as_str(),
        ]);
        solver_table_builder.push_record([
            "Tolerance",
            format!("{:.0E}", self.solver.settings.tolerance).as_str(),
        ]);
        solver_table_builder.push_record([
            "Picard Damping",
            self.solver.settings.picard_damping.to_string().as_str(),
        ]);
        match &self.solver.settings.mdiis_settings {
            Some(settings) => {
                solver_table_builder
                    .push_record(["MDIIS Damping", settings.damping.to_string().as_str()]);
                solver_table_builder.push_record(["Depth", settings.depth.to_string().as_str()])
            }
            None => (),
        }
        match &self.solver.settings.gillan_settings {
            Some(settings) => solver_table_builder
                .push_record(["Num. Basis", settings.nbasis.to_string().as_str()]),
            None => (),
        }

        let solver_table = solver_table_builder.build().to_string();

        println!("{}", solver_table);
    }

    fn build_potential(&mut self, problem: &mut DataRs) {
        let potential = Potential::new(&self.potential);
        // set up total potential
        let npts = problem.grid.npts;
        let num_sites_a = problem.data_a.borrow().sites.len();
        let num_sites_b = problem.data_b.borrow().sites.len();
        let shape = (npts, num_sites_a, num_sites_b);
        let (mut u_nb, mut u_c) = (Array::zeros(shape), Array::zeros(shape));
        // compute nonbonded interaction
        (potential.nonbonded)(
            &problem.data_a.borrow().sites,
            &problem.data_b.borrow().sites,
            &problem.grid.rgrid,
            &mut u_nb,
        );
        // compute electrostatic interaction
        (potential.coulombic)(
            &problem.data_a.borrow().sites,
            &problem.data_b.borrow().sites,
            &problem.grid.rgrid,
            &mut u_c,
        );
        u_c *= problem.system.amph;
        // set total interaction
        problem.interactions.ur = u_nb + u_c;

        // compute renormalised potentials
        (potential.renormalisation_real)(
            &problem.data_a.borrow().sites,
            &problem.data_b.borrow().sites,
            &problem.grid.rgrid,
            &mut problem.interactions.ur_lr,
        );
        (potential.renormalisation_fourier)(
            &problem.data_a.borrow().sites,
            &problem.data_b.borrow().sites,
            &problem.grid.kgrid,
            &mut problem.interactions.uk_lr,
        );

        problem.interactions.ur_lr *= problem.system.amph;
        problem.interactions.uk_lr *= problem.system.amph;

        // set short range interactions
        problem.interactions.u_sr = &problem.interactions.ur - &problem.interactions.ur_lr;
    }

    fn build_intramolecular_correlation(&mut self, problem: &mut DataRs) {
        let distances_a = Self::distance_matrix(
            &problem.data_a.borrow().species,
            &problem.data_a.borrow().species,
        );
        let distances_b = Self::distance_matrix(
            &problem.data_b.borrow().species,
            &problem.data_b.borrow().species,
        );

        Self::intramolecular_corr_impl(
            &distances_a,
            &problem.grid.kgrid,
            &mut problem.data_a.borrow_mut().wk,
        );
        Self::intramolecular_corr_impl(
            &distances_b,
            &problem.grid.kgrid,
            &mut problem.data_b.borrow_mut().wk,
        );
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
