#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    OutOfMemoryRisk,
    OutOfMemory,
    KernelLaunchFailure,
    DeviceUnavailable,
    TransferFailure,
    Unknown,
}
