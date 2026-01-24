#![allow(dead_code)]

use crate::v16::executor::execution_context::{ExecutionContext, RuntimeFacade};

/// Isolated context used for speculative execution.
#[derive(Debug)]
pub struct SpeculativeContext<R: RuntimeFacade> {
    pub label: String,
    pub context: ExecutionContext<R>,
}

impl<R: RuntimeFacade> SpeculativeContext<R> {
    pub fn new(label: String, context: ExecutionContext<R>) -> Self {
        Self { label, context }
    }
}
