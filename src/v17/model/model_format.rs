#![allow(dead_code)]

/// Supported model formats for Atenia.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFormat {
    Onnx,
    SafeTensors,
    Gguf,
    Raw,
}
