pub use crate::tensor::Device;

pub fn device_name(dev: Device) -> &'static str {
    match dev {
        Device::CPU => "CPU",
        Device::GPU => "GPU",
    }
}
