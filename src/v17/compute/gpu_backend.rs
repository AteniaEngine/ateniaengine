#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::guards::guard_action::GuardAction;
use crate::v17::loader::model_loader::LoadedModelHandle;

use super::backend_trait::ComputeBackend;
use super::compute_errors::ComputeError;
use super::cpu_backend::CpuBackend;
use super::ops::{add, matmul, relu};
use super::tensor::Tensor;

/// Minimal GPU backend abstraction. In this version it delegates to the CPU
/// backend to ensure determinism and CI-safety, but keeps a separate type so
/// that real GPU implementations can be plugged in behind feature flags.
#[derive(Debug, Clone)]
pub struct GpuBackend {
    inner: CpuBackend,
}

impl GpuBackend {
    pub fn new() -> Self {
        Self {
            inner: CpuBackend::new(),
        }
    }

    /// Run inference using the same contract and guard semantics as
    /// `CpuBackend::run_inference`, currently delegating entirely to the CPU
    /// backend. This preserves determinism while providing a GPU-shaped API.
    pub fn run_inference(
        &self,
        model: &LoadedModelHandle,
        input: &Tensor,
        contract: &ExecutionContract,
        guard_action: GuardAction,
    ) -> Result<Tensor, ComputeError> {
        self.inner.run_inference(model, input, contract, guard_action)
    }
}

impl ComputeBackend for GpuBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, ComputeError> {
        matmul(a, b)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, ComputeError> {
        add(a, b)
    }

    fn relu(&self, x: &Tensor) -> Result<Tensor, ComputeError> {
        Ok(relu(x))
    }
}
