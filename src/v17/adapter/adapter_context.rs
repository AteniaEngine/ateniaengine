#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::guards::guard_action::GuardAction;
use crate::v17::compute::tensor::Tensor;
use crate::v17::loader::model_loader::LoadedModelHandle;

/// Ephemeral context used by the execution adapter during a single run.
#[derive(Debug)]
pub struct AdapterContext {
    pub model: LoadedModelHandle,
    pub contract: ExecutionContract,
    pub guard_action: GuardAction,
    pub executed_steps: Vec<usize>,
    pub last_output: Option<Tensor>,
    pub aborted: bool,
}

impl AdapterContext {
    pub fn new(model: LoadedModelHandle, contract: ExecutionContract, guard_action: GuardAction) -> Self {
        Self {
            model,
            contract,
            guard_action,
            executed_steps: Vec::new(),
            last_output: None,
            aborted: false,
        }
    }
}
