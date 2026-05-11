#![allow(dead_code)]

use super::loader_errors::LoaderError;

/// Simple policies governing how and when a model may be loaded into memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderPolicy {
    /// Load the entire model as long as there is enough available RAM.
    LoadAll,
    /// Fail fast if available RAM is strictly less than required.
    FailIfInsufficientRam,
}

impl LoaderPolicy {
    pub fn check_memory(&self, required: u64, available: u64) -> Result<(), LoaderError> {
        if available < required {
            match self {
                LoaderPolicy::LoadAll | LoaderPolicy::FailIfInsufficientRam => {
                    Err(LoaderError::InsufficientMemory {
                        required,
                        available,
                    })
                }
            }
        } else {
            Ok(())
        }
    }
}
