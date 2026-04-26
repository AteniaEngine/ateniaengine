//! TinyLlama-1.1B (HuggingFace Llama-family) model module.
//!
//! Sub-modules:
//! - [`config`]: parsing and validation of `config.json`.
//!
//! Future additions for M4.5-b1 (Paso 3):
//! - `builder`: graph construction from a config.

pub mod builder;
pub mod config;
pub mod weight_loading;

pub use builder::{build_tinyllama, TinyLlamaHandles, TinyLlamaRuntime};
pub use config::{ConfigError, TinyLlamaConfig};
pub use weight_loading::tinyllama_weight_mapper;
