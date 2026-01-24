#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::planner::execution_plan::ExecutionPlan;
use crate::v17::adapter::adapter_context::AdapterContext;
use crate::v17::compute::cpu_backend::CpuBackend;
use crate::v17::loader::model_loader::LoadedModelHandle;
use crate::v17::model::model_artifact::ModelArtifact;

/// Ephemeral context for a single end-to-end inference run.
#[derive(Debug)]
pub struct InferenceContext {
    pub artifact: ModelArtifact,
    pub loaded_model: LoadedModelHandle,
    pub contract: ExecutionContract,
    pub plan: ExecutionPlan,
    pub backend: CpuBackend,
    pub adapter_ctx: AdapterContext,
}
