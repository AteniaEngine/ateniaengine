use crate::tensor::Tensor;
use super::forecaster::MemoryForecaster;
use super::offloading::{OffloadHandle, Offloader};

#[derive(Debug)]
pub struct ManagedTensor {
    pub tensor: Option<Tensor>,
    pub offload: Option<OffloadHandle>,
}

pub struct MemoryManager {
    pub limit_bytes: usize,
    pub safety_margin_bytes: usize,
    pub forecaster: MemoryForecaster,
    pub offloader: Offloader,
}

impl MemoryManager {
    pub fn new(limit_bytes: usize, safety_margin_bytes: usize, disk_directory: String) -> Self {
        Self {
            limit_bytes,
            safety_margin_bytes,
            forecaster: MemoryForecaster::new(),
            offloader: Offloader::new(disk_directory),
        }
    }

    /// Estimate total bytes for the given tensors (ignoring those already offloaded).
    pub fn estimate_total_bytes(&mut self, tensors: &[ManagedTensor]) -> usize {
        self.forecaster.current_bytes = 0;
        for mt in tensors {
            if let Some(ref t) = mt.tensor {
                self.forecaster.register_tensor(t);
            }
        }
        self.forecaster.current_bytes
    }

    pub fn available_bytes(&self) -> usize {
        self.limit_bytes.saturating_sub(self.safety_margin_bytes)
    }

    pub fn is_over_limit(&mut self, tensors: &[ManagedTensor]) -> bool {
        let total = self.estimate_total_bytes(tensors);
        total > self.available_bytes()
    }

    /// Find index of the largest in-memory tensor by estimated_bytes().
    fn index_of_largest_tensor(&self, tensors: &[ManagedTensor]) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_size: usize = 0;

        for (i, mt) in tensors.iter().enumerate() {
            if let Some(ref t) = mt.tensor {
                let size = t.estimated_bytes();
                if size > best_size {
                    best_size = size;
                    best_idx = Some(i);
                }
            }
        }

        best_idx
    }

    /// Ensure that the tensors fit into the memory limit.
    /// If over limit, offload the largest tensor to disk.
    /// Returns the index of the tensor that was offloaded, if any.
    pub fn enforce_limit(&mut self, tensors: &mut [ManagedTensor]) -> Option<usize> {
        if !self.is_over_limit(tensors) {
            return None;
        }

        if let Some(idx) = self.index_of_largest_tensor(tensors) {
            if let Some(t) = tensors[idx].tensor.take() {
                let handle = self.offloader.to_disk(&t);
                tensors[idx].offload = Some(handle);
                return Some(idx);
            }
        }

        None
    }

    /// Reload a previously offloaded tensor from disk back into RAM.
    /// The caller must pass the original shape.
    pub fn load_from_disk(&self, mt: &mut ManagedTensor, shape: Vec<usize>) {
        if let Some(ref handle) = mt.offload {
            let restored = self.offloader.from_disk(handle, shape);
            mt.tensor = Some(restored);
        }
    }
}
