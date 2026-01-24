#![allow(dead_code)]

use crate::v17::snapshot::execution_snapshot::ExecutionSnapshot;

/// Expected behavior baseline for a given execution scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionBaseline {
    pub reference_snapshot: ExecutionSnapshot,
    pub allow_backend_change: bool,
}
