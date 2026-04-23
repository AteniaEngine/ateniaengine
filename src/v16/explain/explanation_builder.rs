#![allow(dead_code)]

use crate::v15::policy::types::DecisionBias;
use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::feedback::execution_event::ExecutionEvent;
use crate::v16::feedback::execution_outcome::ExecutionOutcome;
use crate::v16::guards::guard_action::GuardAction;
use crate::v16::speculative::speculative_plan::SpeculativePlan;

use super::execution_explanation::{ExecutionExplanation, StepExecutionExplanation};
use super::explain_errors::ExplainError;

pub struct ExplanationBuilder;

impl ExplanationBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        decision_bias: &DecisionBias,
        contract: &ExecutionContract,
        plan_summary: String,
        events: Vec<ExecutionEvent>,
        outcome: ExecutionOutcome,
        guard_actions: Vec<(usize, GuardAction)>,
        speculative_plan: Option<SpeculativePlan>,
    ) -> Result<ExecutionExplanation, ExplainError> {
        if events.is_empty() {
            return Err(ExplainError::MissingInformation(
                "No execution events provided".to_string(),
            ));
        }

        let mut logical_ts = 0u64;
        for e in &events {
            if e.logical_timestamp < logical_ts {
                return Err(ExplainError::InconsistentEvents(
                    "Events are not in non-decreasing timestamp order".to_string(),
                ));
            }
            logical_ts = e.logical_timestamp;
        }

        // Build per-step explanations from events and guard actions.
        let mut steps: Vec<StepExecutionExplanation> = Vec::new();

        for (idx, action) in guard_actions {
            let speculative = speculative_plan
                .as_ref()
                .map(|p| idx < p.base_plan.steps.len())
                .unwrap_or(false);

            let description = match action {
                GuardAction::Continue => "step executed under normal conditions".to_string(),
                GuardAction::Degrade => "step executed in degraded mode".to_string(),
                GuardAction::DeepDegrade => {
                    "step executed in deep-degraded mode (disk spillover)".to_string()
                }
                GuardAction::Abort => "step associated with abort decision".to_string(),
            };

            steps.push(StepExecutionExplanation {
                step_index: idx,
                description,
                guard_action: Some(action),
                speculative,
            });
        }

        // Summary derived from outcome kind.
        let summary = match outcome.kind {
            crate::v16::feedback::execution_outcome::ExecutionOutcomeKind::Completed => {
                "Execution completed successfully".to_string()
            }
            crate::v16::feedback::execution_outcome::ExecutionOutcomeKind::Failed => {
                "Execution failed".to_string()
            }
            crate::v16::feedback::execution_outcome::ExecutionOutcomeKind::Aborted => {
                "Execution was aborted".to_string()
            }
            crate::v16::feedback::execution_outcome::ExecutionOutcomeKind::PartiallyCompleted => {
                "Execution partially completed".to_string()
            }
        };

        Ok(ExecutionExplanation {
            summary,
            decision_bias: decision_bias.clone(),
            contract: contract.clone(),
            plan_summary,
            steps,
            events,
            outcome,
            speculative_plan,
        })
    }
}
