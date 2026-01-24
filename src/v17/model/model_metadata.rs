#![allow(dead_code)]

/// Descriptive, immutable metadata about a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub family: String,
    pub author: Option<String>,
    pub checksum: String,
    pub estimated_size_bytes: u64,
}

impl ModelMetadata {
    pub fn new(
        name: String,
        version: String,
        family: String,
        author: Option<String>,
        checksum: String,
        estimated_size_bytes: u64,
    ) -> Self {
        Self {
            name,
            version,
            family,
            author,
            checksum,
            estimated_size_bytes,
        }
    }
}
