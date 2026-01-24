#![allow(dead_code)]

use std::fs;
use std::path::Path;

use crate::v17::model::model_artifact::ModelArtifact;
use crate::v17::model::model_format::ModelFormat;

use super::loader_errors::LoaderError;
use super::loader_policy::LoaderPolicy;
use super::memory_map::{MemoryMap, MemorySegment};

/// Handle representing a model that has been loaded into memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedModelHandle {
    pub artifact_id: String,
    pub format: ModelFormat,
    pub memory_map: MemoryMap,
    /// Raw bytes of the model as loaded into RAM.
    pub bytes: Vec<u8>,
}

pub struct ModelLoader;

impl ModelLoader {
    /// Load a model from disk into memory according to the provided policy and
    /// available RAM. This function does not perform any inference or
    /// interpretation of the model; it only reads bytes.
    pub fn load(
        artifact: &ModelArtifact,
        policy: &LoaderPolicy,
        available_ram_bytes: u64,
    ) -> Result<LoadedModelHandle, LoaderError> {
        policy.check_memory(artifact.total_size_bytes, available_ram_bytes)?;

        let path = Path::new(&artifact.location);
        if !path.exists() {
            return Err(LoaderError::FileNotFound(artifact.location.clone()));
        }

        let metadata = fs::metadata(path).map_err(|e| LoaderError::IoError(e.to_string()))?;
        let actual_size = metadata.len();
        if actual_size != artifact.total_size_bytes {
            return Err(LoaderError::SizeMismatch {
                expected: artifact.total_size_bytes,
                actual: actual_size,
            });
        }

        let bytes = fs::read(path).map_err(|e| LoaderError::IoError(e.to_string()))?;
        if bytes.len() as u64 != artifact.total_size_bytes {
            return Err(LoaderError::SizeMismatch {
                expected: artifact.total_size_bytes,
                actual: bytes.len() as u64,
            });
        }

        let memory_map = MemoryMap {
            artifact_id: artifact.id.clone(),
            total_size_bytes: artifact.total_size_bytes,
            loaded_bytes: artifact.total_size_bytes,
            segments: vec![MemorySegment {
                offset: 0,
                length: artifact.total_size_bytes,
            }],
        };

        Ok(LoadedModelHandle {
            artifact_id: artifact.id.clone(),
            format: artifact.format.clone(),
            memory_map,
            bytes,
        })
    }
}
