#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ManifestError {
    InvalidVersion(String),
    InconsistentCapabilities(String),
}
