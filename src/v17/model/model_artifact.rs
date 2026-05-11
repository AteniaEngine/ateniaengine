#![allow(dead_code)]

use super::model_errors::ModelError;
use super::model_format::ModelFormat;
use super::model_metadata::ModelMetadata;

/// Immutable description of a model artifact known to Atenia.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifact {
    /// Stable identifier for this model within Atenia.
    pub id: String,
    /// Descriptive metadata.
    pub metadata: ModelMetadata,
    /// Format of the underlying model file(s).
    pub format: ModelFormat,
    /// Logical location of the model weights (path or URI), not accessed at runtime here.
    pub location: String,
    /// Declared total size of the model bytes on disk (or remote storage).
    pub total_size_bytes: u64,
}

impl ModelArtifact {
    /// Construct a new `ModelArtifact` from the provided components, performing
    /// basic validation without any IO or runtime interaction.
    pub fn new(
        id: String,
        metadata: ModelMetadata,
        format: ModelFormat,
        location: String,
        total_size_bytes: u64,
    ) -> Result<Self, ModelError> {
        if id.trim().is_empty() {
            return Err(ModelError::InvalidMetadata(
                "id must not be empty".to_string(),
            ));
        }
        if metadata.name.trim().is_empty() {
            return Err(ModelError::InvalidMetadata(
                "metadata.name must not be empty".to_string(),
            ));
        }
        if metadata.version.trim().is_empty() {
            return Err(ModelError::InvalidMetadata(
                "metadata.version must not be empty".to_string(),
            ));
        }
        if metadata.checksum.trim().is_empty() {
            return Err(ModelError::InvalidMetadata(
                "metadata.checksum must not be empty".to_string(),
            ));
        }
        if location.trim().is_empty() {
            return Err(ModelError::InvalidPath(
                "location must not be empty".to_string(),
            ));
        }
        if total_size_bytes == 0 {
            return Err(ModelError::InvalidSize(
                "total_size_bytes must be greater than zero".to_string(),
            ));
        }

        // All `ModelFormat` variants defined here are supported by construction,
        // so there is no need for further runtime checks.

        Ok(Self {
            id,
            metadata,
            format,
            location,
            total_size_bytes,
        })
    }
}
