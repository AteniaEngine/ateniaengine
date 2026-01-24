#![allow(dead_code)]

/// Binary or parametrized capability flags describing what the engine supports.
#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityDescriptor {
    pub supports_gpu_execution: bool,
    pub supports_replay: bool,
    pub supports_abortability: bool,
    pub supports_determinism: bool,
    pub supports_snapshot_sealing: bool,
}
