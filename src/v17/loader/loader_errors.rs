#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderError {
    FileNotFound(String),
    SizeMismatch { expected: u64, actual: u64 },
    InsufficientMemory { required: u64, available: u64 },
    PolicyDenied(String),
    IoError(String),
}
