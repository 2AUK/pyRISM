use crate::data::configuration::{
    operator::OperatorConfig, potential::PotentialConfig, problem::ProblemConfig,
    solver::SolverConfig,
};

pub mod operator;
pub mod potential;
pub mod problem;
pub mod solver;

#[derive(Debug, Clone)]
pub struct Configuration {
    pub data_config: ProblemConfig,
    pub operator_config: OperatorConfig,
    pub potential_config: PotentialConfig,
    pub solver_config: SolverConfig,
}
