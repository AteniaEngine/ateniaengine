#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderError {
    FileNotFound(String),
    SizeMismatch {
        expected: u64,
        actual: u64,
    },
    InsufficientMemory {
        required: u64,
        available: u64,
    },
    PolicyDenied(String),
    IoError(String),
    /// Safetensors header or structure is malformed: invalid JSON,
    /// header_size inconsistent with actual header length, missing
    /// required fields in a tensor entry, shape elements that do not
    /// match the declared data range, etc. The inner string carries a
    /// human-facing explanation of what specifically failed.
    ///
    /// Introduced in M4-a (safetensors reader).
    InvalidFormat(String),
    /// A tensor in the file has a dtype that the current Atenia build
    /// cannot convert to `Vec<f32>`. In M4-a only F32 is supported;
    /// F16, BF16, FP8, and integer dtypes surface through this
    /// variant. M4-d extends support to BF16 and F16 via
    /// host-side downcast on load.
    UnsupportedDType(String),
    /// A tensor from the safetensors file has a shape that does not
    /// match the shape of the parameter node it would be loaded into.
    /// Loading proceeds no further for this tensor; fix the source
    /// checkpoint or the graph architecture so the shapes agree.
    ///
    /// Introduced in M4-c (weight mapper).
    ShapeMismatch {
        tensor_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// **M12.4 H5** — operator-readable rendering. Before this, the only
/// way a `LoaderError` reached a CLI surface was through
/// `PipelineError::Loader`'s `{:?}` debug form, which leaked the
/// raw Rust enum shape (`InvalidFormat("…")`) instead of a sentence.
/// Each variant already carries a human-facing inner string.
impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::FileNotFound(p) => write!(f, "file not found: {p}"),
            LoaderError::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected} bytes, got {actual}")
            }
            LoaderError::InsufficientMemory {
                required,
                available,
            } => write!(
                f,
                "insufficient memory: need {required} bytes, {available} available"
            ),
            LoaderError::PolicyDenied(s) => write!(f, "policy denied: {s}"),
            LoaderError::IoError(s) => write!(f, "io error: {s}"),
            LoaderError::InvalidFormat(s) => write!(f, "invalid format: {s}"),
            LoaderError::UnsupportedDType(s) => write!(f, "unsupported dtype: {s}"),
            LoaderError::ShapeMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch for tensor '{tensor_name}': expected {expected:?}, got {actual:?}"
            ),
        }
    }
}
