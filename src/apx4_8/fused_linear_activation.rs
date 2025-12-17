use crate::tensor::Tensor;
use crate::nn::activations::silu;
use crate::nn::linear::linear;

/// Fused Linear + Activation operation for CPU.
/// For now we support SiLU, which ya existe como activación en el grafo.
pub fn exec_fused_linear_silu(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let lin = linear(x, w, b);
    silu(&lin)
}
