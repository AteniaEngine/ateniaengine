#![allow(dead_code)]

use crate::v16::contract::constraints::ConstraintKind;
use crate::v16::contract::execution_contract::ExecutionContract;

use super::execution_plan::ExecutionPlan;
use super::plan_step::{PlanStep, PlanStepKind};
use super::planner_errors::PlannerError;

pub struct ExecutionPlanner;

impl ExecutionPlanner {
    pub fn build_plan(contract: &ExecutionContract) -> Result<ExecutionPlan, PlannerError> {
        // Basic validation: there must be at least one allowed backend.
        if contract.allowed_backends.is_empty() {
            return Err(PlannerError::InvalidContract(
                "ExecutionContract has no allowed backends".to_string(),
            ));
        }

        // If the contract is internally contradictory (e.g., requires stability but has no
        // stability-oriented constraints), treat it as unplannable. For APX 16.1 we keep
        // this simple: if require_stability is true but there is no RequireStability
        // constraint, we consider it invalid.
        if contract.require_stability {
            let has_require = contract
                .constraints
                .items
                .iter()
                .any(|c| matches!(c.kind, ConstraintKind::RequireStability));

            if !has_require {
                return Err(PlannerError::UnplannableContract(
                    "Contract requires stability but lacks a RequireStability constraint"
                        .to_string(),
                ));
            }
        }

        let mut steps: Vec<PlanStep> = Vec::new();

        // Step 1: ensure memory headroom according to contract constraints.
        steps.push(PlanStep {
            kind: PlanStepKind::EnsureMemoryHeadroom,
            description: "Ensure memory headroom satisfies contract constraints".to_string(),
            preconditions: vec!["Contract invariants hold".to_string()],
            postconditions: vec!["Observed headroom is compatible with contract".to_string()],
            abortable: true,
            requires_verification: true,
        });

        // Step 2: select a backend candidate consistent with allowed/forbidden sets.
        steps.push(PlanStep {
            kind: PlanStepKind::SelectBackendCandidate,
            description: "Select a backend candidate within allowed/forbidden sets".to_string(),
            preconditions: vec!["Memory headroom is acceptable".to_string()],
            postconditions: vec![
                "A backend candidate consistent with contract is identified".to_string(),
            ],
            abortable: true,
            requires_verification: true,
        });

        // Step 3: prepare fallback if required by the contract.
        if contract.require_fallback {
            steps.push(PlanStep {
                kind: PlanStepKind::PrepareFallback,
                description: "Prepare fallback path required by the contract".to_string(),
                preconditions: vec!["Primary backend candidate has been selected".to_string()],
                postconditions: vec![
                    "Fallback path is available if primary path fails".to_string(),
                ],
                abortable: true,
                requires_verification: true,
            });
        }

        // Step 4: mark tensors as conceptually movable if offload or similar behavior is
        // allowed by the contract. This is purely descriptive in APX 16.1.
        steps.push(PlanStep {
            kind: PlanStepKind::MarkTensorsMovable,
            description: "Mark tensors as conceptually movable according to contract".to_string(),
            preconditions: vec![
                "Backend selection and fallback preparation (if any) are complete".to_string(),
            ],
            postconditions: vec![
                "Tensors are considered movable for future execution steps".to_string(),
            ],
            abortable: true,
            requires_verification: true,
        });

        Ok(ExecutionPlan {
            contract: contract.clone(),
            steps,
            globally_abortable: true,
        })
    }
}
