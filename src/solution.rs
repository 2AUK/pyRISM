use crate::data::{Correlations, Interactions};
use crate::operator::OperatorConfig;
use crate::potential::PotentialConfig;
use crate::solver::SolverConfig;

#[derive(Clone, Debug)]
pub struct SolvedData {
    pub solver_config: SolverConfig,
    pub potential_config: PotentialConfig,
    pub operator_config: OperatorConfig,
    pub interactions: Interactions,
    pub correlations: Correlations,
}

impl SolvedData {
    pub fn new(
        solver_config: SolverConfig,
        potential_config: PotentialConfig,
        operator_config: OperatorConfig,
        interactions: Interactions,
        correlations: Correlations,
    ) -> Self {
        SolvedData {
            solver_config,
            potential_config,
            operator_config,
            interactions,
            correlations,
        }
    }
}
