#![allow(dead_code)]

use crate::v16::executor::execution_context::{ExecutionContext, RuntimeFacade};

/// Isolated context used for deterministic replay.
#[derive(Debug)]
pub struct ReplayContext<R: RuntimeFacade> {
    pub label: String,
    pub context: ExecutionContext<R>,
}

impl<R: RuntimeFacade> ReplayContext<R> {
    pub fn new(label: String, context: ExecutionContext<R>) -> Self {
        Self { label, context }
    }
}
