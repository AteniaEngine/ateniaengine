#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    Retry,
    FallbackToCPU,
    MoveTensorToRAM,
    MoveTensorToSSD,
    ReduceBatch,
    SkipKernel,
    Abort,
    None,
}
