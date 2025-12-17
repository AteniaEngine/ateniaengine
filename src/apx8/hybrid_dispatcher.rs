// APX 8.2 — Hybrid Dispatcher
// NO ejecuta GPU real. Usa la estructura DualGraph construida en 8.1.
// CPU siempre es fallback. GPU ejecuta stub equivalente.

use crate::amg::graph::Graph;
use crate::tensor::{Tensor, Device};
use crate::apx8::gpu_transfer_estimator::{GPUTransferEstimator, TransferEstimate};
use crate::apx8::dualgraph::DevicePlacement;
use crate::apx8::gpu_partition::suggest_partition;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecDevice {
    CPU,
    GPU,
}

pub struct HybridDispatcher;

impl HybridDispatcher {
    pub fn select_device(op: &str) -> ExecDevice {
        // Por ahora, reglas fijas (APX 8.3 empieza a mejorarla)
        match op {
            "MatMul" => ExecDevice::GPU,
            "Linear" => ExecDevice::GPU,
            _ => ExecDevice::CPU,
        }
    }

    /// APX 8.3: estimador cuantitativo de dispositivo para un tensor concreto.
    /// No mueve datos ni ejecuta kernels reales; sólo usa GPUTransferEstimator.
    pub fn choose_device_for(t: &Tensor) -> ExecDevice {
        // Fallback obligatorio a CPU cuando el modo aún no es 8.3.
        if !crate::apx_mode_at_least("8.3") {
            return ExecDevice::CPU;
        }

        // Si el tensor ya está en GPU o el entorno no es adecuado, mantenemos CPU.
        if matches!(t.device, Device::GPU) {
            return ExecDevice::CPU;
        }

        let placement = DevicePlacement::CPU;
        let _est: TransferEstimate = GPUTransferEstimator::estimate(t, placement);

        // Heurística 8.3: usar un umbral simple en número de elementos.
        //  - Tensores pequeños: coste de copia domina ⇒ CPU.
        //  - Tensores grandes: se prepara para GPU ⇒ GPU.
        let elems = t.num_elements();
        let threshold: usize = 256 * 256; // 65_536 elementos

        if elems >= threshold {
            ExecDevice::GPU
        } else {
            ExecDevice::CPU
        }
    }

    pub fn exec_cpu(graph: &mut Graph, node_id: usize, record_tape: bool) {
        graph.execute_single_inner(node_id, record_tape);
    }

    pub fn exec_gpu_stub(graph: &mut Graph, node_id: usize, record_tape: bool) {
        // Stub GPU (APX 8.2 NO ejecuta kernels)
        graph.execute_single_inner(node_id, record_tape);
    }

    pub fn dispatch(graph: &mut Graph, node_id: usize, op: &str, record_tape: bool) {
        match Self::select_device(op) {
            ExecDevice::CPU => Self::exec_cpu(graph, node_id, record_tape),
            ExecDevice::GPU => {
                if crate::apx_mode_at_least("8.2") {
                    Self::exec_gpu_stub(graph, node_id, record_tape)
                } else {
                    Self::exec_cpu(graph, node_id, record_tape)
                }
            }
        }
    }

    /// APX 8.19: estrategia GPU simulada basada en la forma de un IR/tensor.
    /// No toca ejecución real; sólo devuelve una descripción textual.
    pub fn choose_gpu_strategy(shape: &[usize]) -> String {
        let plan = suggest_partition(shape);
        format!("GPU_STRATEGY({:?})", plan.policy)
    }
}
