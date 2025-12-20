use crate::tensor::{Tensor, Device};
use crate::cuda::{self, matmul::cuda_matmul, fused_linear_silu::cuda_fused_linear_silu};
use crate::amg::graph::Graph;
use crate::apx4_12::pool_dispatcher::try_gpu_with_pool;

fn apx_trace_enabled() -> bool {
    matches!(std::env::var("APX_TRACE").as_deref(), Ok("1")) && !crate::apx_is_silent()
}

/// Simple heuristic to decide whether it is worth using the CUDA MatMul kernel.
pub fn gpu_can_run_matmul(m: usize, k: usize, n: usize) -> bool {
    // Require a minimum amount of work to amortize host<->device overhead.
    let ops = m.saturating_mul(k).saturating_mul(n);
    if ops <= 256 {
        return false;
    }

    // The remaining validations (device, dtype, shapes) are performed in `try_gpu_matmul`.
    if !cuda::cuda_available() {
        return false;
    }

    true
}

/// Attempt to execute MatMul on GPU.
/// Returns `true` if it ran on GPU and `out` contains the result.
pub fn try_gpu_matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> bool {
    // Only supports 2D f32 tensors on logical CPU.
    if a.device != Device::CPU || b.device != Device::CPU {
        return false;
    }
    if a.dtype != b.dtype || a.dtype != out.dtype {
        return false;
    }
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return false;
    }

    let m = a.shape[0];
    let k = a.shape[1];
    if b.shape[0] != k {
        return false;
    }
    let n = b.shape[1];

    if out.shape != [m, n] {
        return false;
    }

    if !gpu_can_run_matmul(m, k, n) {
        return false;
    }

    if !cuda::cuda_available() {
        return false;
    }

    // APX 4.12: use the MemoryPool dispatcher to decide whether to execute
    // on GPU or let the caller fall back to the CPU path.
    let mut ran_gpu = false;
    let bytes_needed = m
        .saturating_mul(n)
        .saturating_mul(std::mem::size_of::<f32>());

    try_gpu_with_pool(
        bytes_needed,
        || {
            let gpu_out = cuda_matmul(a, b, m, k, n);
            if gpu_out.shape == out.shape {
                out.data.clone_from_slice(&gpu_out.data);
                ran_gpu = true;
                if apx_trace_enabled() {
                    println!("[APX 4.11] GPU MatMul executed");
                }
            }
        },
        || {
            // CPU fallback: do nothing here; the caller will see `false`
            // and execute the standard CPU path.
        },
    );

    ran_gpu
}

/// Attempt to execute Linear on GPU: y = x·w + b (optional).
/// Returns `true` if it ran on GPU and `out` contains the result.
pub fn try_gpu_linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>, out: &mut Tensor) -> bool {
    // APX 4.11 MiniFlux: fully disable the GPU path for Linear (with or without
    // bias). Always execute on CPU to correctly record backward and avoid
    // divergences between kernels.
    let _ = (x, w, b, out); // avoid warnings for unused parameters
    return false;

    // Original code intentionally disabled.
}

/// APX 4.13: fused Linear+SiLU execution hook from node IDs in the graph.
/// For simplicity, we only support the case where bias is present; if there is
/// no bias, we fall back to the standard CPU path (Linear + SiLU).
pub unsafe fn fused_linear_silu_gpu(
    x_id: usize,
    w_id: usize,
    b_id: Option<usize>,
    out_id: usize,
    graph: &mut Graph,
    _record_tape: bool,
) {
    let x = graph.nodes[x_id]
        .output
        .as_ref()
        .expect("fused_linear_silu: missing x output")
        .clone();
    let w = graph.nodes[w_id]
        .output
        .as_ref()
        .expect("fused_linear_silu: missing w output")
        .clone();

    let (m, k) = (x.shape[0], x.shape[1]);
    let n = w.shape[1];

    let mut out = Tensor::with_layout(
        vec![m, n],
        0.0,
        x.device,
        crate::tensor::Layout::Contiguous,
        x.dtype,
    );

    if let Some(bid) = b_id {
        let b = graph.nodes[bid]
            .output
            .as_ref()
            .expect("fused_linear_silu: missing bias output")
            .clone();

        cuda_fused_linear_silu(
            &x.data,
            &w.data,
            &b.data,
            &mut out.data,
            m,
            k,
            n,
        );
        graph.nodes[out_id].set_output(out);
    } else {
        // No bias: use the standard CPU Linear + SiLU path.
        let mut tmp = crate::nn::linear::linear(&x, &w, None);
        tmp = crate::nn::activations::silu(&tmp);
        graph.nodes[out_id].set_output(tmp);
    }
}
