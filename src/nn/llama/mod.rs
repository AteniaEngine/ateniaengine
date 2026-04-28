//! Llama-family model module (TinyLlama 1.1B, SmolLM2, Llama 3.x,
//! Qwen 2.5, …). All checkpoints sharing the HuggingFace
//! `LlamaForCausalLM` architecture compose from the same
//! [`config`], [`builder`], and [`weight_loading`] primitives;
//! per-model variations (tied embeddings, eps, rope theta) ride
//! in the config and are honored by the builder.
//!
//! Sub-modules:
//! - [`config`]: parsing and validation of `config.json`.
//! - [`builder`]: graph construction from a config.
//! - [`weight_loading`]: Llama-family weight transforms and
//!   parameter mapping into the AMG.

pub mod builder;
pub mod config;
pub mod weight_loading;

pub use builder::{build_llama, LlamaHandles, LlamaRuntime};
pub use config::{ConfigError, LlamaConfig};
pub use weight_loading::llama_weight_mapper;
