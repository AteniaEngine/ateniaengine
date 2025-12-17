use crate::tensor::Device;

#[derive(Debug, Clone)]
pub struct DeviceContext {
    pub device: Device,
}

impl DeviceContext {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.device, Device::CPU)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self.device, Device::GPU)
    }
}
