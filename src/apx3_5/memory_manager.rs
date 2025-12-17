use crate::tensor::{Device, Tensor};

pub struct MemoryManager;

impl MemoryManager {
    pub fn move_tensor(t: &Tensor, dev: Device) -> Tensor {
        match dev {
            Device::CPU => t.to_cpu(),
            Device::GPU => t.to_gpu(),
        }
    }
}
