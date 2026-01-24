#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    InvalidMetadata(String),
    UnsupportedFormat(String),
    InvalidPath(String),
    InvalidSize(String),
}
