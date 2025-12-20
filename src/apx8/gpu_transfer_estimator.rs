// APX 8.3 – GPU Transfer Estimator (GTE)
// Does NOT execute real CUDA. Does NOT move real tensors.
// Only estimates costs. 100% safe and deterministic.

use crate::tensor::Tensor;
use crate::apx8::dualgraph::DevicePlacement;

#[derive(Debug, Clone)]
pub struct TransferEstimate {
    pub h2d_ms: f64,
    pub d2h_ms: f64,
    pub stay_gpu_ms: f64,
}

pub struct GPUTransferEstimator;

impl GPUTransferEstimator {
    pub fn estimate(_t: &Tensor, _placement: DevicePlacement) -> TransferEstimate {
        let bytes = (_t.num_elements() as f64) * (_t.dtype.size_in_bytes() as f64);

        // Default calibrated values
        let bw_pcie_gb_s = 12.0;     // typical PCIe 4.0
        let gpu_exec_base_ms = 0.02; // minimal GPU overhead

        let h2d_ms = bytes / (bw_pcie_gb_s * 1e9) * 1000.0;
        let d2h_ms = h2d_ms * 1.05;  // return is slightly more expensive
        let stay_gpu_ms = gpu_exec_base_ms;

        TransferEstimate {
            h2d_ms,
            d2h_ms,
            stay_gpu_ms,
        }
    }
}
