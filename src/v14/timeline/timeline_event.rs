#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineEvent {
    KernelStart {
        kernel_id: String,
        device: String,
    },
    KernelEnd {
        kernel_id: String,
        device: String,
    },
    DeviceSelected {
        device: String,
    },
    MemoryTransfer {
        src_device: String,
        dst_device: String,
        bytes: u64,
    },
}
