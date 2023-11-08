use crate::data::{
    configuration::{
        Configuration,
        {
            operator::OperatorConfig, potential::PotentialConfig, problem::ProblemConfig,
            solver::SolverConfig,
        },
    },
    core::{Correlations, Interactions},
};

use serde::{Deserialize, Serialize};

pub struct JobDetails {}

pub struct Solutions {
    pub config: Configuration,
    pub vv: SolvedData,
    pub uv: Option<SolvedData>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SolvedData {
    pub data_config: ProblemConfig,
    pub solver_config: SolverConfig,
    pub potential_config: PotentialConfig,
    pub operator_config: OperatorConfig,
    pub interactions: Interactions,
    pub correlations: Correlations,
}

impl SolvedData {
    pub fn new(
        data_config: ProblemConfig,
        solver_config: SolverConfig,
        potential_config: PotentialConfig,
        operator_config: OperatorConfig,
        interactions: Interactions,
        correlations: Correlations,
    ) -> Self {
        SolvedData {
            data_config,
            solver_config,
            potential_config,
            operator_config,
            interactions,
            correlations,
        }
    }
}
