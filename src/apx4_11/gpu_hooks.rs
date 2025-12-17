use crate::tensor::{Tensor, Device};
use crate::cuda::{self, matmul::cuda_matmul, fused_linear_silu::cuda_fused_linear_silu};
use crate::amg::graph::Graph;
use crate::apx4_12::pool_dispatcher::try_gpu_with_pool;

fn apx_trace_enabled() -> bool {
    matches!(std::env::var("APX_TRACE").as_deref(), Ok("1")) && !crate::apx_is_silent()
}

/// Heurística simple para decidir si merece la pena usar el kernel CUDA de MatMul.
pub fn gpu_can_run_matmul(m: usize, k: usize, n: usize) -> bool {
    // Requerir un mínimo de trabajo para compensar el overhead host<->device.
    let ops = m.saturating_mul(k).saturating_mul(n);
    if ops <= 256 {
        return false;
    }

    // El resto de validaciones (device, dtype, shapes) se harán en `try_gpu_matmul`.
    if !cuda::cuda_available() {
        return false;
    }

    true
}

/// Intenta ejecutar MatMul en GPU.
/// Devuelve `true` si se ejecutó en GPU y `out` contiene el resultado.
pub fn try_gpu_matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> bool {
    // Sólo soportamos tensores 2D f32 en CPU lógico.
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

    // APX 4.12: usar el dispatcher con MemoryPool para decidir si ejecutar
    // en GPU o dejar que el caller caiga a la ruta CPU.
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
            // CPU fallback: no hacemos nada aquí; el caller verá `false`
            // y ejecutará la ruta CPU estándar.
        },
    );

    ran_gpu
}

/// Intenta ejecutar Linear en GPU: y = x·w + b (opcional).
/// Devuelve `true` si se ejecutó en GPU y `out` contiene el resultado.
pub fn try_gpu_linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>, out: &mut Tensor) -> bool {
    // APX 4.11 MiniFlux: deshabilitar completamente la ruta GPU para Linear
    // (tanto con bias como sin bias). Ejecutamos siempre en CPU para
    // registrar correctamente el backward y evitar divergencias entre
    // kernels.
    let _ = (x, w, b, out); // evitar warnings por parámetros no usados
    return false;

    // Código original desactivado intencionalmente.
}

/// APX 4.13: hook de ejecución fusionada Linear+SiLU a partir de IDs de nodos
/// en el grafo. Para simplificar, sólo soportamos el caso con bias presente;
/// si no hay bias, caemos a la ruta CPU estándar (Linear + SiLU).
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
        // Sin bias: usar la ruta CPU Linear + SiLU estándar.
        let mut tmp = crate::nn::linear::linear(&x, &w, None);
        tmp = crate::nn::activations::silu(&tmp);
        graph.nodes[out_id].set_output(tmp);
    }
}
