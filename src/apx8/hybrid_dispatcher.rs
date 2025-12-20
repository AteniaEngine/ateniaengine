// APX 8.2 — Hybrid Dispatcher
// Does NOT execute real GPU. Uses the DualGraph structure built in 8.1.
// CPU is always the fallback. GPU runs an equivalent stub.

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
        // For now, fixed rules (APX 8.3 starts improving this)
        match op {
            "MatMul" => ExecDevice::GPU,
            "Linear" => ExecDevice::GPU,
            _ => ExecDevice::CPU,
        }
    }

    /// APX 8.3: quantitative device estimator for a specific tensor.
    /// Does not move data nor execute real kernels; it only uses GPUTransferEstimator.
    pub fn choose_device_for(t: &Tensor) -> ExecDevice {
        // Mandatory fallback to CPU when mode is not yet 8.3.
        if !crate::apx_mode_at_least("8.3") {
            return ExecDevice::CPU;
        }

        // If the tensor is already on GPU or the environment is not suitable, keep CPU.
        if matches!(t.device, Device::GPU) {
            return ExecDevice::CPU;
        }

        let placement = DevicePlacement::CPU;
        let _est: TransferEstimate = GPUTransferEstimator::estimate(t, placement);

        // 8.3 heuristic: use a simple threshold on number of elements.
        //  - Small tensors: copy cost dominates => CPU.
        //  - Large tensors: prepare for GPU => GPU.
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
        // GPU stub (APX 8.2 does NOT execute kernels)
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

    /// APX 8.19: simulated GPU strategy based on an IR/tensor shape.
    /// Does not touch real execution; only returns a textual description.
    pub fn choose_gpu_strategy(shape: &[usize]) -> String {
        let plan = suggest_partition(shape);
        format!("GPU_STRATEGY({:?})", plan.policy)
    }
}
