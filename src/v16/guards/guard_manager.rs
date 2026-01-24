#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;

use super::execution_guard::ExecutionGuard;
use super::guard_action::GuardAction;
use super::guard_conditions::GuardConditions;
use super::guard_errors::GuardError;

pub struct GuardManager {
    guards: Vec<Box<dyn ExecutionGuard>>, 
}

impl GuardManager {
    pub fn new(guards: Vec<Box<dyn ExecutionGuard>>) -> Self {
        Self { guards }
    }

    /// Evaluate all guards and produce a single recommended action.
    ///
    /// Abort dominates over Degrade, which dominates over Continue. The
    /// resulting recommendation is also checked against the execution contract
    /// for basic legal constraints.
    pub fn evaluate(
        &self,
        contract: &ExecutionContract,
        conditions: &GuardConditions,
    ) -> Result<GuardAction, GuardError> {
        let mut final_action = GuardAction::Continue;

        for guard in &self.guards {
            let action = guard.evaluate(contract, conditions);

            // Abort dominates everything.
            if matches!(final_action, GuardAction::Abort) || matches!(action, GuardAction::Abort)
            {
                final_action = GuardAction::Abort;
                continue;
            }

            // Degrade dominates Continue.
            if matches!(final_action, GuardAction::Degrade)
                || matches!(action, GuardAction::Degrade)
            {
                final_action = GuardAction::Degrade;
                continue;
            }

            // Otherwise both are Continue; keep Continue.
            final_action = GuardAction::Continue;
        }

        // Basic legality check: continuing under a clear pre-OOM signal while
        // the contract requires stability is considered illegal.
        if matches!(final_action, GuardAction::Continue)
            && conditions.pre_oom_signal
            && contract.require_stability
        {
            return Err(GuardError::IllegalAction(
                "Cannot continue under pre-OOM conditions with required stability".to_string(),
            ));
        }

        Ok(final_action)
    }
}
