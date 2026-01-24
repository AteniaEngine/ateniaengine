#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;

use super::guard_action::GuardAction;
use super::guard_conditions::GuardConditions;

/// Pure, evaluative guard that inspects conditions and recommends an action.
pub trait ExecutionGuard: Send + Sync {
    fn name(&self) -> &'static str;

    fn evaluate(&self, contract: &ExecutionContract, conditions: &GuardConditions) -> GuardAction;
}
