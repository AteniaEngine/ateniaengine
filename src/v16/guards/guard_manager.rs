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
    /// Dominance ordering (M3-e.11.5):
    ///
    ///   Abort > DeepDegrade > Degrade > Continue
    ///
    /// where each level wins over the ones to its right. The
    /// combination rule is "highest-severity across the guards":
    /// if any guard emits `Abort`, the verdict is `Abort`;
    /// otherwise if any emits `DeepDegrade`, the verdict is
    /// `DeepDegrade`; otherwise if any emits `Degrade`, the verdict
    /// is `Degrade`; otherwise `Continue`.
    ///
    /// The resulting recommendation is also checked against the
    /// execution contract for basic legal constraints.
    pub fn evaluate(
        &self,
        contract: &ExecutionContract,
        conditions: &GuardConditions,
    ) -> Result<GuardAction, GuardError> {
        // Rank-based aggregation: compute the severity of every
        // guard's action as an integer and keep the maximum. This
        // replaces the cascading matches! pattern that was hard to
        // extend when `DeepDegrade` landed between `Degrade` and
        // `Abort`.
        //
        // Ranks (higher = more severe):
        //   Continue     = 0
        //   Degrade      = 1
        //   DeepDegrade  = 2
        //   Abort        = 3
        fn rank(a: &GuardAction) -> u8 {
            match a {
                GuardAction::Continue => 0,
                GuardAction::Degrade => 1,
                GuardAction::DeepDegrade => 2,
                GuardAction::Abort => 3,
            }
        }

        let mut final_action = GuardAction::Continue;
        for guard in &self.guards {
            let action = guard.evaluate(contract, conditions);
            if rank(&action) > rank(&final_action) {
                final_action = action;
            }
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
