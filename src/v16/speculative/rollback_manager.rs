#![allow(dead_code)]

/// Manages rollback for speculative execution by keeping an original snapshot
/// of the runtime facade.
#[derive(Debug, Clone, PartialEq)]
pub struct RollbackManager<R: Clone> {
    original_runtime: R,
}

impl<R: Clone> RollbackManager<R> {
    pub fn new(runtime: &R) -> Self {
        Self {
            original_runtime: runtime.clone(),
        }
    }

    pub fn rollback(&self, runtime: &mut R) {
        *runtime = self.original_runtime.clone();
    }
}
