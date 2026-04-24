#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderError {
    FileNotFound(String),
    SizeMismatch { expected: u64, actual: u64 },
    InsufficientMemory { required: u64, available: u64 },
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
