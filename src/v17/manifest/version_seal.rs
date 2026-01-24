#![allow(dead_code)]

use super::engine_manifest::EngineManifest;
use crate::v17::snapshot::snapshot_hash::hash_str;
use super::manifest_errors::ManifestError;

/// Deterministic fingerprint of an engine manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct VersionSeal {
    pub manifest_hash: String,
}

impl VersionSeal {
    pub fn from_manifest(manifest: &EngineManifest) -> Result<Self, ManifestError> {
        if manifest.engine_version.trim().is_empty() {
            return Err(ManifestError::InvalidVersion(
                "engine_version must not be empty".to_string(),
            ));
        }

        if manifest.learning_enabled {
            return Err(ManifestError::InconsistentCapabilities(
                "learning_enabled must be false for APX 17".to_string(),
            ));
        }

        let hash_input = manifest.to_json();
        let manifest_hash = hash_str(&hash_input);
        Ok(Self { manifest_hash })
    }
}
