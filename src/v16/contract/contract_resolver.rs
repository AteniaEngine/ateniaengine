#![allow(dead_code)]

use crate::v15::policy::types::DecisionBias;

use super::contract_errors::ContractError;
use super::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use super::execution_contract::{ExecutionBackend, ExecutionContract};

pub struct ContractResolver;

impl ContractResolver {
    pub fn resolve_contract(
        bias: &DecisionBias,
        state: &RuntimeState,
    ) -> Result<ExecutionContract, ContractError> {
        // Invariant checks: bias must be normalized, memory_headroom in [0, 1].
        if !bias.is_normalized() {
            return Err(ContractError::InvariantViolation(
                "DecisionBias must be normalized".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&state.memory_headroom) {
            return Err(ContractError::InvariantViolation(
                "memory_headroom must be in [0.0, 1.0]".to_string(),
            ));
        }

        // Derive a simple aggressiveness score from the bias.
        let base_aggressiveness = (bias.latency_weight + bias.risk_weight).min(1.0);

        let mut constraints_items: Vec<Constraint> = Vec::new();

        // Hard constraint: if there is essentially no headroom and the bias is very aggressive,
        // consider this incompatible.
        if state.memory_headroom < 0.05 && base_aggressiveness > 0.8 {
            return Err(ContractError::IncompatibleIntent(
                "Highly aggressive bias with almost no headroom".to_string(),
            ));
        }

        // Require stability when the system is unstable or has recovered recently, or when
        // the bias strongly favors stability.
        let require_stability = state.recent_recovery || !state.is_stable || bias.stability_weight >= 0.8;
        if require_stability {
            constraints_items.push(Constraint::hard(ConstraintKind::RequireStability));
        }

        // Limit aggressiveness when headroom is low or the system is unstable.
        let mut max_aggressiveness = base_aggressiveness;
        if state.memory_headroom < 0.5 || !state.is_stable {
            max_aggressiveness = max_aggressiveness.min(0.5);
            constraints_items.push(Constraint::soft(ConstraintKind::LimitAggressiveness { max: 0.5 }));
        }

        // Memory headroom constraint (hard if extremely low, soft otherwise).
        if state.memory_headroom < 0.2 {
            constraints_items.push(Constraint::hard(ConstraintKind::MemoryHeadroom { min: state.memory_headroom }));
        } else {
            constraints_items.push(Constraint::soft(ConstraintKind::MemoryHeadroom { min: state.memory_headroom }));
        }

        // Backend constraints.
        let mut allowed_backends = vec![ExecutionBackend::Local];
        let mut forbidden_backends: Vec<ExecutionBackend> = Vec::new();
        let mut require_fallback = false;

        if state.offload_supported {
            // Interpret lower offload_cost_weight as more willingness to use offload.
            if bias.offload_cost_weight <= 0.5 {
                allowed_backends.push(ExecutionBackend::Offload);
                require_fallback = true;
                constraints_items.push(Constraint::soft(ConstraintKind::RequireFallback));
            } else {
                forbidden_backends.push(ExecutionBackend::Offload);
                constraints_items.push(Constraint::hard(ConstraintKind::ForbidOffload));
            }
        } else {
            forbidden_backends.push(ExecutionBackend::Offload);
            constraints_items.push(Constraint::hard(ConstraintKind::ForbidOffload));
        }

        // Final derived constraints collection.
        let constraints = Constraints {
            items: constraints_items,
        };

        if allowed_backends.is_empty() {
            return Err(ContractError::NoLegalExecution(
                "No allowed backends under current constraints".to_string(),
            ));
        }

        Ok(ExecutionContract {
            bias: bias.clone(),
            runtime_snapshot: state.clone(),
            allowed_backends,
            forbidden_backends,
            max_aggressiveness,
            require_fallback,
            require_stability,
            constraints,
        })
    }
}
