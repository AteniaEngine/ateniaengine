#![allow(dead_code)]

/// Minimal trait that represents the capabilities the safe executor needs from
/// the underlying runtime. Implementations may be real runtimes or test mocks.
pub trait RuntimeFacade {
    fn ensure_memory_headroom(&mut self) -> Result<(), String>;
    fn select_backend_candidate(&mut self) -> Result<(), String>;
    fn prepare_fallback(&mut self) -> Result<(), String>;
    fn mark_tensors_movable(&mut self) -> Result<(), String>;
}

/// ExecutionContext wraps a `RuntimeFacade` implementation and is the only way
/// the executor layer interacts with the runtime.
#[derive(Debug)]
pub struct ExecutionContext<R: RuntimeFacade> {
    pub runtime: R,
}

impl<R: RuntimeFacade> ExecutionContext<R> {
    pub fn new(runtime: R) -> Self {
        Self { runtime }
    }
}
