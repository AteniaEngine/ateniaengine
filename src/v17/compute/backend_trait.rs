#![allow(dead_code)]

use super::compute_errors::ComputeError;
use super::tensor::Tensor;

/// Common contract for compute backends used by Atenia.
pub trait ComputeBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, ComputeError>;
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, ComputeError>;
    fn relu(&self, x: &Tensor) -> Result<Tensor, ComputeError>;
}
