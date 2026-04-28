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
    /// Per-head dimension when the config sets it explicitly
    /// (Llama 3.2 onwards). When absent, the effective head_dim is
    /// `hidden_size / num_attention_heads` — see
    /// [`Self::effective_head_dim`]. Architectures like Phi-3 medium
    /// where the two diverge will need the explicit field; for the
    /// Llama family seen so far the two coincide.
    pub head_dim: Option<usize>,
    /// `rope_scaling` block. `None` for plain RoPE (TinyLlama,
    /// SmolLM2, Qwen 2.5). Currently the only recognised
    /// `rope_type` is `"llama3"` (Llama 3.x). Anything else parses
    /// to `None` with a tracing-friendly warning at parse time.
    pub rope_scaling: Option<RopeScaling>,
}

/// Parsed `rope_scaling` block from a Llama-family `config.json`.
///
/// Variants intentionally narrow: only the schemas Atenia knows how
/// to honour at runtime are accepted. Future scaling families
/// (`"yarn"`, `"longrope"`, ...) are added as new variants when we
/// add support, not parsed-but-ignored.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    /// Llama 3 piecewise inverse-frequency scaling (Llama 3.1, 3.2,
    /// 3.3). Algorithm: see
    /// `huggingface/transformers::modeling_rope_utils::_compute_llama3_parameters`.
    Llama3 {
        /// Long-context expansion factor (32.0 for Llama 3.2).
        factor: f32,
        /// Lower-bound frequency multiplier; controls where the
        /// smooth-interp band ends on the short-wavelength side.
        low_freq_factor: f32,
        /// Upper-bound frequency multiplier; mirrors
        /// `low_freq_factor` on the long-wavelength side.
        high_freq_factor: f32,
        /// Pre-training context length used as the wavelength
        /// reference (8192 for Llama 3.2).
        original_max_position_embeddings: u32,
    },
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

fn get_optional_usize(v: &Value, key: &str) -> Result<Option<usize>, ConfigError> {
    match v.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(other) => other
            .as_u64()
            .map(|n| Some(n as usize))
            .ok_or_else(|| {
                ConfigError::Parse(format!(
                    "field `{}` is not a non-negative integer",
                    key
                ))
            }),
    }
}

/// Parse the `rope_scaling` JSON object if present and recognised.
///
/// Tolerates two field-name conventions for the discriminator:
/// `"rope_type"` (transformers >=4.43) and `"type"` (legacy /
/// pre-4.43). Returns `Ok(None)` when the field is missing, JSON
/// null, or carries an unsupported `rope_type` — Atenia only
/// recognises `"llama3"` today, so any other value is silently
/// downgraded to no-scaling rather than failing the whole parse.
fn get_rope_scaling(v: &Value) -> Result<Option<RopeScaling>, ConfigError> {
    let block = match v.get("rope_scaling") {
        None | Some(Value::Null) => return Ok(None),
        Some(other) => other,
    };
    if !block.is_object() {
        return Err(ConfigError::Parse(
            "field `rope_scaling` must be a JSON object".into(),
        ));
    }
    // Accept both `rope_type` (modern) and `type` (legacy).
    let rope_type = block
        .get("rope_type")
        .or_else(|| block.get("type"))
        .and_then(|x| x.as_str());
    match rope_type {
        Some("llama3") => {
            let factor = block
                .get("factor")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| {
                    ConfigError::Parse(
                        "rope_scaling.llama3 requires numeric `factor`".into(),
                    )
                })? as f32;
            let low_freq_factor = block
                .get("low_freq_factor")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| {
                    ConfigError::Parse(
                        "rope_scaling.llama3 requires numeric `low_freq_factor`".into(),
                    )
                })? as f32;
            let high_freq_factor = block
                .get("high_freq_factor")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| {
                    ConfigError::Parse(
                        "rope_scaling.llama3 requires numeric `high_freq_factor`".into(),
                    )
                })? as f32;
            let original_max_position_embeddings = block
                .get("original_max_position_embeddings")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| {
                    ConfigError::Parse(
                        "rope_scaling.llama3 requires integer `original_max_position_embeddings`"
                            .into(),
                    )
                })? as u32;
            Ok(Some(RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            }))
        }
        // Unknown / unsupported scaling — treat as no-scaling.
        // Future variants (yarn, longrope, dynamic) will land as
        // explicit branches before this catch-all.
        _ => Ok(None),
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
            head_dim: get_optional_usize(&v, "head_dim")?,
            rope_scaling: get_rope_scaling(&v)?,
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
    ///
    /// Kept for backward compatibility. New call sites should prefer
    /// [`Self::effective_head_dim`], which honours the explicit
    /// `head_dim` field set by Llama 3.2 and friends. For all
    /// Llama-family checkpoints seen so far the two values coincide.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Effective per-head dimension. Explicit `head_dim` from the
    /// config wins when present; otherwise falls back to
    /// `hidden_size / num_attention_heads`. Llama 3.2 sets the
    /// field explicitly, even though the two values agree (64).
    /// Future architectures (e.g. Phi-3 medium) may decouple them.
    pub fn effective_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
    }

    /// Convenience accessor for the parsed `rope_scaling` block.
    /// Returns `None` for plain-RoPE checkpoints
    /// (TinyLlama, SmolLM2, Qwen 2.5). Returns `Some(&RopeScaling)`
    /// for Llama 3.x.
    pub fn effective_rope_scaling(&self) -> Option<&RopeScaling> {
        self.rope_scaling.as_ref()
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
