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
pub mod builder_shared;
pub mod config;
pub mod generator;
pub mod pipeline;
pub mod weight_loading;

pub use builder::{build_llama, LlamaHandles, LlamaRuntime};
pub use builder_shared::{build_llama_with_store, LlamaHandlesShared, BuildError};
pub use config::{ConfigError, LlamaConfig, RopeScaling};
pub use generator::{
    generate_greedy, CollectingTokenSink, GenerateError, GeneratedToken,
    GenerationConfig, StdoutTokenSink, TokenSink,
};
pub use pipeline::{GenerationPipeline, PipelineError};
pub use weight_loading::llama_weight_mapper;
