#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::guards::guard_action::GuardAction;
use crate::v17::loader::model_loader::LoadedModelHandle;

use super::backend_trait::ComputeBackend;
use super::compute_errors::ComputeError;
use super::ops::{add, matmul, relu};
use super::tensor::Tensor;

/// Minimal CPU-only backend for executing simple inference.
#[derive(Debug, Clone)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    /// Run a minimal inference: apply a single linear layer (matrix multiply)
    /// followed by ReLU. This is intentionally simple and CPU-only.
    pub fn run_inference(
        &self,
        model: &LoadedModelHandle,
        input: &Tensor,
        contract: &ExecutionContract,
        guard_action: GuardAction,
    ) -> Result<Tensor, ComputeError> {
        // Respect contract: if stability is required but runtime is not stable,
        // refuse to run.
        if contract.require_stability && !contract.runtime_snapshot.is_stable {
            return Err(ComputeError::ContractViolation(
                "runtime is not stable under contract".to_string(),
            ));
        }

        // Respect guard decision.
        match guard_action {
            GuardAction::Abort => {
                return Err(ComputeError::AbortedByGuard(
                    "execution aborted by guard".to_string(),
                ))
            }
            // Degrade and DeepDegrade both mean "keep running in a
            // reduced-capacity mode". v17's CPU backend does not
            // act on the distinction (it already runs on host
            // memory; spillover-to-disk semantics are a higher-
            // level concern); they collapse to the same Continue-
            // like branch here.
            GuardAction::Degrade
            | GuardAction::DeepDegrade
            | GuardAction::Continue => {}
        }

        // Interpret the model bytes as a very small dense layer: we expect a
        // square matrix of size n x n stored as f32 in little-endian form.
        let n = input.shape.iter().product::<usize>();
        let expected_bytes = (n * n * 4) as u64;
        if model.bytes.len() as u64 != expected_bytes {
            return Err(ComputeError::ContractViolation(
                "model bytes do not match expected simple square weight matrix".to_string(),
            ));
        }

        let mut weights = Vec::with_capacity(n * n);
        for chunk in model.bytes.chunks_exact(4) {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            weights.push(f32::from_le_bytes(bytes));
        }

        let w = Tensor::new(vec![n, n], weights)
            .map_err(|e| ComputeError::ShapeMismatch(format!("weights: {e}")))?;

        let y = matmul(&w, input)?;
        Ok(relu(&y))
    }
}

impl ComputeBackend for CpuBackend {
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
