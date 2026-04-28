//! Parsing, validation, and derived helpers for a Llama-family
//! `config.json` (HuggingFace convention). Covers TinyLlama 1.1B,
//! SmolLM2, Llama 3.x, Qwen 2.5, and other `LlamaForCausalLM`
//! variants.
//!
//! Scope (M4.5-b1, Paso 1): only the fields actually consumed by the
//! graph builder land here. Tokenizer-only and training-only fields
//! (`hidden_act`, `initializer_range`, `pretraining_tp`, `use_cache`,
//! `architectures`, `model_type`, `transformers_version`, `torch_dtype`,
//! `rope_scaling`) are intentionally ignored — adding them later is
//! cheap and non-breaking thanks to `serde_json::Value` being open by
//! default.

use std::fmt;
use std::fs;
use std::path::Path;

use serde_json::Value;

/// Configuration for a Llama-family model.
///
/// Constructed from a HuggingFace `config.json` via
/// [`LlamaConfig::from_json_file`] or
/// [`LlamaConfig::from_json_str`].
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Number of K/V heads. If `< num_attention_heads`, the model uses
    /// Grouped Query Attention (GQA). For TinyLlama-1.1B-Chat-v1.0
    /// this is 4 vs 32 Q heads → GQA factor 8.
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    /// `rope_theta` from the config, stored as `u32` to match the
    /// `NodeType::RoPE { base_freq: u32 }` API. The config encodes
    /// it as a float (`10000.0`) but no Llama-family model uses a
    /// non-integer base frequency.
    pub rope_theta: u32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    /// Whether `q_proj`/`k_proj`/`v_proj`/`o_proj` carry a bias.
    /// `Some(false)` for TinyLlama and SmolLM2 (explicit in
    /// `config.json`); `None` for Qwen 2.5 (the field is absent
    /// from the official config — Qwen2 hard-codes QKV biases on,
    /// see [`Self::effective_attention_bias`]).
    pub attention_bias: Option<bool>,
    /// HuggingFace `model_type` discriminator: `"llama"`,
    /// `"qwen2"`, etc. Optional because some legacy configs omit
    /// it. Used by [`Self::effective_attention_bias`] to resolve
    /// per-family defaults when the explicit field is absent.
    pub model_type: Option<String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    /// `pad_token_id` is absent from many Llama configs (TinyLlama
    /// included), so it is optional.
    pub pad_token_id: Option<u32>,
}

/// Errors that can arise while loading or validating a Llama-family config.
#[derive(Debug)]
pub enum ConfigError {
    /// The config file could not be read from disk.
    Io(std::io::Error),
    /// The config could not be parsed as JSON or a required field was
    /// missing / had the wrong shape.
    Parse(String),
    /// The parsed config violated a structural constraint
    /// (e.g. `hidden_size % num_attention_heads != 0`).
    Validation(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "config IO error: {}", e),
            ConfigError::Parse(s) => write!(f, "config parse error: {}", s),
            ConfigError::Validation(s) => write!(f, "config validation error: {}", s),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        ConfigError::Io(e)
    }
}

// --- JSON field extraction helpers ---------------------------------------

fn get_u64(v: &Value, key: &str) -> Result<u64, ConfigError> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .ok_or_else(|| ConfigError::Parse(format!("missing or non-integer field `{}`", key)))
}

fn get_usize(v: &Value, key: &str) -> Result<usize, ConfigError> {
    Ok(get_u64(v, key)? as usize)
}

fn get_u32(v: &Value, key: &str) -> Result<u32, ConfigError> {
    let n = get_u64(v, key)?;
    if n > u32::MAX as u64 {
        return Err(ConfigError::Parse(format!(
            "field `{}` value {} exceeds u32::MAX",
            key, n
        )));
    }
    Ok(n as u32)
}

fn get_f32(v: &Value, key: &str) -> Result<f32, ConfigError> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| ConfigError::Parse(format!("missing or non-numeric field `{}`", key)))
        .map(|f| f as f32)
}

fn get_bool(v: &Value, key: &str) -> Result<bool, ConfigError> {
    v.get(key)
        .and_then(|x| x.as_bool())
        .ok_or_else(|| ConfigError::Parse(format!("missing or non-bool field `{}`", key)))
}

fn get_optional_bool(v: &Value, key: &str) -> Result<Option<bool>, ConfigError> {
    match v.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(other) => other
            .as_bool()
            .ok_or_else(|| ConfigError::Parse(format!("field `{}` is not a bool", key)))
            .map(Some),
    }
}

fn get_optional_string(v: &Value, key: &str) -> Result<Option<String>, ConfigError> {
    match v.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(other) => other
            .as_str()
            .ok_or_else(|| ConfigError::Parse(format!("field `{}` is not a string", key)))
            .map(|s| Some(s.to_string())),
    }
}

fn get_optional_u32(v: &Value, key: &str) -> Result<Option<u32>, ConfigError> {
    match v.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(other) => match other.as_u64() {
            Some(n) if n <= u32::MAX as u64 => Ok(Some(n as u32)),
            _ => Err(ConfigError::Parse(format!(
                "field `{}` is not a non-negative integer fitting in u32",
                key
            ))),
        },
    }
}

/// `rope_theta` may be encoded as a float (`10000.0`) or as an integer
/// (`10000`). Accept both, fail if non-integer-valued or negative.
fn get_rope_theta(v: &Value) -> Result<u32, ConfigError> {
    let raw = v
        .get("rope_theta")
        .ok_or_else(|| ConfigError::Parse("missing field `rope_theta`".into()))?;
    let f = raw
        .as_f64()
        .ok_or_else(|| ConfigError::Parse("`rope_theta` is not numeric".into()))?;
    if !f.is_finite() || f < 0.0 || f > u32::MAX as f64 {
        return Err(ConfigError::Parse(format!(
            "`rope_theta` value {} out of u32 range",
            f
        )));
    }
    if f.fract() != 0.0 {
        return Err(ConfigError::Parse(format!(
            "`rope_theta` must be integer-valued, got {}",
            f
        )));
    }
    Ok(f as u32)
}

// --- impl ----------------------------------------------------------------

impl LlamaConfig {
    /// Parse from a JSON string. Validates after parsing.
    pub fn from_json_str(s: &str) -> Result<Self, ConfigError> {
        let v: Value = serde_json::from_str(s)
            .map_err(|e| ConfigError::Parse(format!("JSON syntax error: {}", e)))?;

        let cfg = LlamaConfig {
            vocab_size: get_usize(&v, "vocab_size")?,
            hidden_size: get_usize(&v, "hidden_size")?,
            num_hidden_layers: get_usize(&v, "num_hidden_layers")?,
            num_attention_heads: get_usize(&v, "num_attention_heads")?,
            num_key_value_heads: get_usize(&v, "num_key_value_heads")?,
            intermediate_size: get_usize(&v, "intermediate_size")?,
            max_position_embeddings: get_usize(&v, "max_position_embeddings")?,
            rope_theta: get_rope_theta(&v)?,
            rms_norm_eps: get_f32(&v, "rms_norm_eps")?,
            tie_word_embeddings: get_bool(&v, "tie_word_embeddings")?,
            attention_bias: get_optional_bool(&v, "attention_bias")?,
            model_type: get_optional_string(&v, "model_type")?,
            bos_token_id: get_u32(&v, "bos_token_id")?,
            eos_token_id: get_u32(&v, "eos_token_id")?,
            pad_token_id: get_optional_u32(&v, "pad_token_id")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read and parse a config from a file path.
    pub fn from_json_file(path: &Path) -> Result<Self, ConfigError> {
        let s = fs::read_to_string(path)?;
        Self::from_json_str(&s)
    }

    /// Per-head dimension: `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// K/V heads share the same per-head dimension as Q heads in
    /// standard Llama-family GQA.
    pub fn kv_head_dim(&self) -> usize {
        self.head_dim()
    }

    /// GQA grouping factor: how many Q heads share a single K/V head.
    /// For standard MHA (no GQA) this is 1.
    pub fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Resolved value of `attention_bias` for graph construction.
    ///
    /// HuggingFace `config.json` semantics differ across Llama-family
    /// architectures: Llama and SmolLM explicitly emit
    /// `"attention_bias": false`, but Qwen2 omits the field entirely
    /// and hard-codes QKV biases on inside `Qwen2Attention`. This
    /// helper centralises the disambiguation:
    ///
    /// 1. If the config carries an explicit `attention_bias`, that
    ///    value wins.
    /// 2. Otherwise, the `model_type` drives the default: `"qwen2"`
    ///    → `true`; everything else → `false`.
    pub fn effective_attention_bias(&self) -> bool {
        if let Some(explicit) = self.attention_bias {
            return explicit;
        }
        matches!(self.model_type.as_deref(), Some("qwen2"))
    }

    /// Rough parameter-count estimate. Used by tests to sanity-check
    /// against the real ~1.1B figure for TinyLlama; not exact (drops
    /// LayerNorm scales and the final LM-head bias if absent).
    pub fn total_params_estimate(&self) -> usize {
        let h = self.hidden_size;
        let i = self.intermediate_size;
        let n_q = self.num_attention_heads;
        let n_kv = self.num_key_value_heads;
        let head_dim = self.head_dim();
        let l = self.num_hidden_layers;
        let v = self.vocab_size;

        let embed = v * h;
        let q = (n_q * head_dim) * h;
        let k = (n_kv * head_dim) * h;
        let v_proj = (n_kv * head_dim) * h;
        let o = h * h;
        let attn_per_layer = q + k + v_proj + o;
        let mlp_per_layer = 3 * (h * i); // gate + up + down
        let norms_per_layer = 2 * h; // input + post_attn
        let per_layer = attn_per_layer + mlp_per_layer + norms_per_layer;

        let final_norm = h;
        let lm_head = if self.tie_word_embeddings { 0 } else { v * h };

        embed + per_layer * l + final_norm + lm_head
    }

    /// Structural validation. Called from both
    /// [`Self::from_json_str`] and direct constructors so that an
    /// invalid config never escapes the loader.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_hidden_layers == 0
        {
            return Err(ConfigError::Validation(
                "vocab/hidden/heads/layers must all be positive".into(),
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ConfigError::Validation(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(ConfigError::Validation(format!(
                "num_attention_heads ({}) must be a multiple of num_key_value_heads ({}) (GQA)",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        let head_dim = self.head_dim();
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(ConfigError::Validation(format!(
                "head_dim ({}) must be positive and even (RoPE half-split requirement)",
                head_dim
            )));
        }
        if self.rms_norm_eps <= 0.0 || !self.rms_norm_eps.is_finite() {
            return Err(ConfigError::Validation(format!(
                "rms_norm_eps must be a positive finite number, got {}",
                self.rms_norm_eps
            )));
        }
        Ok(())
    }
}
