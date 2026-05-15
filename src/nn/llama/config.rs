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
    /// **M11.C** — Gemma 2 attention-logit soft-cap (`50.0` for
    /// Gemma 2 2B/9B/27B). When `Some(c)`, the attention scores
    /// pre-softmax are passed through `c * tanh(x / c)`. `None`
    /// for every non-Gemma-2 family (Llama, Mistral, Qwen, Phi-3,
    /// Falcon3 — none of them apply attention-logit soft-cap).
    pub attn_logit_softcapping: Option<f32>,
    /// **M11.C** — Gemma 2 final-logit soft-cap (`30.0` for
    /// Gemma 2). When `Some(c)`, the LM-head output is passed
    /// through `c * tanh(x / c)` before sampling. `None` for
    /// every non-Gemma-2 family.
    pub final_logit_softcapping: Option<f32>,
    /// **M11.C** — Gemma 2 sliding-window length (`4096` for
    /// Gemma 2). When `Some(w)`, alternating layers (in Gemma 2:
    /// the even-indexed ones) restrict attention to the last `w`
    /// tokens. `None` for full-attention-only architectures.
    pub sliding_window: Option<u32>,
    /// **M11.C** — Gemma 2 attention-scale denominator
    /// (`256` for Gemma 2 2B; the Q vectors are pre-divided by
    /// `sqrt(query_pre_attn_scalar)` instead of the usual
    /// `sqrt(head_dim)`). For Gemma 2 2B these coincide
    /// (`head_dim = 256`), but the field is read explicitly so
    /// future Gemma variants where they diverge work without
    /// schema changes. `None` falls back to `sqrt(head_dim)`.
    /// Encoded in the config as an integer (`256`); read as
    /// `f32` for the downstream `1/sqrt(.)` computation.
    pub query_pre_attn_scalar: Option<f32>,
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
    /// **M11.B** — Microsoft Phi-3 / Phi-3.5 LongRope scaling.
    ///
    /// Per-dimension scaling factors that swap based on the
    /// observed sequence length:
    ///
    /// - When `seq_len <= original_max_position_embeddings`, the
    ///   inverse frequency at index `i` is divided by
    ///   `short_factor[i]`.
    /// - When `seq_len > original_max_position_embeddings`, the
    ///   inverse frequency at index `i` is divided by
    ///   `long_factor[i]`.
    ///
    /// Both factor vectors must have length `head_dim / 2`.
    ///
    /// Additionally, when scaling is active, `cos`/`sin` outputs
    /// are multiplied by an `attention_factor` derived from the
    /// `max_position_embeddings / original_max_position_embeddings`
    /// ratio:
    ///
    /// ```text
    ///   scale = max_position_embeddings / original_max_position_embeddings
    ///   attention_factor = sqrt(1 + ln(scale) / ln(original_max_position_embeddings))
    ///                          when scale > 1.0
    ///                      else 1.0
    /// ```
    ///
    /// Reference:
    /// `huggingface/transformers::modeling_rope_utils::_compute_longrope_parameters`.
    LongRope {
        /// Per-dimension factor vector applied when
        /// `seq_len <= original_max_position_embeddings`. Length
        /// must equal `head_dim / 2`.
        short_factor: Vec<f32>,
        /// Per-dimension factor vector applied when
        /// `seq_len > original_max_position_embeddings`. Length
        /// must equal `head_dim / 2`.
        long_factor: Vec<f32>,
        /// Pre-training context length (4096 for Phi-3.5 Mini).
        /// Drives both the short/long factor switch and the
        /// `attention_factor` derivation.
        original_max_position_embeddings: u32,
        /// Configured max-context length (131072 for Phi-3.5
        /// Mini). Used together with
        /// `original_max_position_embeddings` to derive the
        /// `attention_factor`.
        max_position_embeddings: u32,
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

/// Strict scalar-`u32` reader. Currently unused at the call-site
/// (M11.A.1 swapped both `bos_token_id` and `eos_token_id` to
/// [`get_u32_or_first_of_array`] for Llama 3.x compatibility).
/// Kept reachable so a future config field that requires strict
/// scalar semantics has the helper available; flipping to
/// `#[allow(dead_code)]` instead of deletion preserves the
/// type-checked behaviour for that future caller.
#[allow(dead_code)]
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

/// **M11.A.1** — read a scalar `u32` field that may also appear as
/// a non-empty array of integers; the first element is returned.
///
/// Llama 3.x ships `eos_token_id` as an array of three sentinel
/// tokens — `[end_of_text, end_of_message, eot_id]` — to support
/// the multi-turn chat protocol where any of the three terminates
/// generation. Atenia's downstream consumers (`generator::is_eos`)
/// compare against a single `u32` today; the first array entry is
/// the canonical end-of-generation token (`128001` =
/// `<|end_of_text|>` for Llama 3.1 / 3.2). A future M11.A.x can
/// extend the contract to honour every sentinel — that is a
/// generator-side change, not a config-side change.
///
/// Returns:
/// - `Ok(n)` when the field is a scalar `u32`.
/// - `Ok(arr[0])` when the field is a non-empty array of `u32`.
/// - `Err` when the field is missing, a non-integer scalar, an
///   empty array, or an array containing non-integer entries.
fn get_u32_or_first_of_array(v: &Value, key: &str) -> Result<u32, ConfigError> {
    let raw = v
        .get(key)
        .ok_or_else(|| ConfigError::Parse(format!("missing field `{}`", key)))?;
    if let Some(n) = raw.as_u64() {
        if n > u32::MAX as u64 {
            return Err(ConfigError::Parse(format!(
                "field `{}` value {} exceeds u32::MAX",
                key, n
            )));
        }
        return Ok(n as u32);
    }
    if let Some(arr) = raw.as_array() {
        if arr.is_empty() {
            return Err(ConfigError::Parse(format!(
                "field `{}` is an empty array (expected ≥ 1 token id)",
                key
            )));
        }
        let first = arr[0].as_u64().ok_or_else(|| {
            ConfigError::Parse(format!(
                "field `{}` is an array but the first element is not an integer",
                key
            ))
        })?;
        if first > u32::MAX as u64 {
            return Err(ConfigError::Parse(format!(
                "field `{}` first array element {} exceeds u32::MAX",
                key, first
            )));
        }
        return Ok(first as u32);
    }
    Err(ConfigError::Parse(format!(
        "field `{}` must be an integer or a non-empty integer array",
        key
    )))
}

fn get_f32(v: &Value, key: &str) -> Result<f32, ConfigError> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| ConfigError::Parse(format!("missing or non-numeric field `{}`", key)))
        .map(|f| f as f32)
}

/// Strict scalar-`bool` reader. **M11.C** swapped the
/// `tie_word_embeddings` call-site to [`get_tie_word_embeddings`]
/// (family-aware default for Gemma 2's omitted field). Kept
/// reachable behind `#[allow(dead_code)]` so a future config
/// field that requires hard-fail-on-missing semantics has a
/// type-checked helper available.
#[allow(dead_code)]
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
        Some(other) => other.as_u64().map(|n| Some(n as usize)).ok_or_else(|| {
            ConfigError::Parse(format!("field `{}` is not a non-negative integer", key))
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
                    ConfigError::Parse("rope_scaling.llama3 requires numeric `factor`".into())
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
        // **M11.B** — Phi-3 / Phi-3.5 LongRope scaling.
        Some("longrope") => {
            let short_factor = parse_f32_array(block, "short_factor", "rope_scaling.longrope")?;
            let long_factor = parse_f32_array(block, "long_factor", "rope_scaling.longrope")?;
            // `original_max_position_embeddings` and
            // `max_position_embeddings` are top-level fields on
            // the Phi-3 config, not inside the rope_scaling
            // block. Read them from the outer config `v`.
            let original_max_position_embeddings = v
                .get("original_max_position_embeddings")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| {
                    ConfigError::Parse(
                        "rope_scaling.longrope requires top-level integer \
                         `original_max_position_embeddings`"
                            .into(),
                    )
                })? as u32;
            let max_position_embeddings = v
                .get("max_position_embeddings")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| {
                    ConfigError::Parse(
                        "rope_scaling.longrope requires top-level integer \
                         `max_position_embeddings`"
                            .into(),
                    )
                })? as u32;
            Ok(Some(RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
                max_position_embeddings,
            }))
        }
        // Unknown / unsupported scaling — treat as no-scaling.
        // Future variants (yarn, dynamic) will land as
        // explicit branches before this catch-all.
        _ => Ok(None),
    }
}

/// Parse a JSON array field into a `Vec<f32>`. Used by the
/// LongRope branch to read the per-dimension factor vectors.
/// Errors with a clear context-prefixed message when the field
/// is missing, not an array, or contains a non-numeric element.
fn parse_f32_array(block: &Value, key: &str, context: &str) -> Result<Vec<f32>, ConfigError> {
    let arr = block
        .get(key)
        .and_then(|x| x.as_array())
        .ok_or_else(|| ConfigError::Parse(format!("{context} requires array field `{key}`")))?;
    arr.iter()
        .enumerate()
        .map(|(i, x)| {
            x.as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| ConfigError::Parse(format!("{context}.{key}[{i}] is not a number")))
        })
        .collect()
}

/// **M11.C** — read an optional `f32` field that may be encoded
/// as either a JSON float (`50.0`) or a JSON integer (`256`).
/// Gemma 2's `query_pre_attn_scalar` ships as an integer in the
/// reference HuggingFace config, while `attn_logit_softcapping`
/// and `final_logit_softcapping` ship as floats. A single helper
/// covers both shapes and rejects null / non-numeric values
/// with a clear error.
fn get_optional_f32(v: &Value, key: &str) -> Result<Option<f32>, ConfigError> {
    match v.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(other) => other.as_f64().map(|f| Some(f as f32)).ok_or_else(|| {
            ConfigError::Parse(format!(
                "field `{}` is not a number (expected float or integer)",
                key
            ))
        }),
    }
}

/// **M11.C** — read `tie_word_embeddings` with a family-aware
/// default for missing fields. Gemma 2's official HuggingFace
/// config omits the field entirely; the upstream `Gemma2Config`
/// defaults it to `True`. Every other Llama-family checkpoint
/// emits the field explicitly (Llama, Mistral, Qwen, Phi-3,
/// Falcon3 all serialise it), so the missing-field branch fires
/// only for `gemma`/`gemma2` today. Other families with the
/// field absent still trigger the original "missing field"
/// parse error to preserve fail-loud behaviour for unexpected
/// config shapes.
///
/// **Phase 12.2** — the family-aware default is now sourced
/// from the adapter registry via
/// [`crate::model_adapters::ConfigPolicy::default_tie_word_embeddings`]
/// instead of a hard-coded `match model_type` here. The
/// behaviour is unchanged: `Gemma2Adapter` returns
/// `Some(true)`; every other adapter inherits the trait
/// default `None`, which produces the same hard-error this
/// function returned pre-Phase-12.2.
fn get_tie_word_embeddings(v: &Value) -> Result<bool, ConfigError> {
    if let Some(explicit) = get_optional_bool(v, "tie_word_embeddings")? {
        return Ok(explicit);
    }
    let model_type = v.get("model_type").and_then(|x| x.as_str());
    let metadata = crate::model_adapters::model_metadata_from_parts(
        None,
        model_type,
        crate::model_adapters::ModelFormat::HfSafetensors,
    );
    match crate::model_adapters::resolve_adapter(&metadata)
        .and_then(|adapter| adapter.default_tie_word_embeddings())
    {
        Some(default) => Ok(default),
        None => Err(ConfigError::Parse(
            "missing or non-bool field `tie_word_embeddings`".into(),
        )),
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

/// Default `rope_theta` for Llama-family architectures that
/// predate the explicit field in `config.json`. **Llama 2** (June
/// 2023) shipped before HuggingFace started serialising `rope_theta`
/// — its `config.json` carries no such field, but the upstream
/// PyTorch implementation hard-codes `theta = 10000.0` inside
/// `LlamaRotaryEmbedding`. Every later checkpoint (TinyLlama,
/// SmolLM2, Qwen 2.5, Llama 3.x) writes the field explicitly, so
/// this default is exercised exclusively by the Llama 1/2 era.
///
/// Defined as a constant so the M4.7.6.a parser change has a
/// single, documented source of truth and so a future
/// architectural family with a different default (none known
/// today) becomes a one-line audit point.
pub const ROPE_THETA_LEGACY_DEFAULT: u32 = 10_000;

/// `rope_theta` may be encoded as a float (`10000.0`) or as an integer
/// (`10000`). Accept both, fail if non-integer-valued or negative.
///
/// **M4.7.6.a**: when the field is absent or `null`, fall back to
/// [`ROPE_THETA_LEGACY_DEFAULT`] (10000) — Llama 2 era checkpoints
/// did not serialise the field. Pre-M4.7.6.a this raised
/// `Parse("missing field rope_theta")`, which prevented Llama 2
/// from loading even though the math was correct.
fn get_rope_theta(v: &Value) -> Result<u32, ConfigError> {
    let raw = match v.get("rope_theta") {
        None => return Ok(ROPE_THETA_LEGACY_DEFAULT),
        Some(Value::Null) => return Ok(ROPE_THETA_LEGACY_DEFAULT),
        Some(x) => x,
    };
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
            tie_word_embeddings: get_tie_word_embeddings(&v)?,
            attention_bias: get_optional_bool(&v, "attention_bias")?,
            model_type: get_optional_string(&v, "model_type")?,
            bos_token_id: get_u32_or_first_of_array(&v, "bos_token_id")?,
            // **M11.A.1** — Llama 3.x ships eos_token_id as a
            // 3-element array of sentinel tokens. Take the first
            // (canonical end-of-text); see helper docstring.
            eos_token_id: get_u32_or_first_of_array(&v, "eos_token_id")?,
            pad_token_id: get_optional_u32(&v, "pad_token_id")?,
            head_dim: get_optional_usize(&v, "head_dim")?,
            rope_scaling: get_rope_scaling(&v)?,
            // **M11.C** — Gemma 2 fields. All `None` for every
            // non-Gemma-2 checkpoint, so existing parsers stay
            // byte-equivalent on Llama / Mistral / Qwen / Phi-3 /
            // Falcon3 fixtures.
            attn_logit_softcapping: get_optional_f32(&v, "attn_logit_softcapping")?,
            final_logit_softcapping: get_optional_f32(&v, "final_logit_softcapping")?,
            sliding_window: get_optional_u32(&v, "sliding_window")?,
            query_pre_attn_scalar: get_optional_f32(&v, "query_pre_attn_scalar")?,
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
    /// 2. Otherwise, the adapter registry resolves the family-specific
    ///    default via
    ///    [`crate::model_adapters::ConfigPolicy::default_attention_bias`]:
    ///    `Qwen2Adapter` returns `Some(true)`; every other adapter
    ///    inherits the trait default `None`, which here translates
    ///    to `false`.
    ///
    /// **Phase 12.3** — pre-Phase-12.3 this method matched
    /// `model_type == "qwen2"` directly. The registry lookup
    /// preserves the same behaviour while keeping family-specific
    /// decisions inside the adapter layer.
    pub fn effective_attention_bias(&self) -> bool {
        if let Some(explicit) = self.attention_bias {
            return explicit;
        }
        let metadata = crate::model_adapters::model_metadata_from_parts(
            None,
            self.model_type.as_deref(),
            crate::model_adapters::ModelFormat::HfSafetensors,
        );
        crate::model_adapters::resolve_adapter(&metadata)
            .and_then(|adapter| adapter.default_attention_bias())
            .unwrap_or(false)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Minimal Llama-style config covering every required field so
    /// that the parser focuses on the field under test.
    fn base_config_value(eos_field: serde_json::Value) -> serde_json::Value {
        json!({
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": eos_field,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 256,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.46.1",
            "vocab_size": 32_000
        })
    }

    /// **M11.A.1 test** — scalar `eos_token_id` (Llama 1 / 2 /
    /// SmolLM2 / Qwen / Mistral / Falcon3 shape) parses unchanged.
    #[test]
    fn eos_token_id_scalar_parses() {
        let v = base_config_value(json!(2));
        let cfg = LlamaConfig::from_json_str(&v.to_string()).expect("scalar eos parses");
        assert_eq!(cfg.eos_token_id, 2);
    }

    /// **M11.A.1 test** — array `eos_token_id` (Llama 3.x shape)
    /// parses; the first element is taken as the canonical EOS.
    #[test]
    fn eos_token_id_array_takes_first() {
        let v = base_config_value(json!([128001, 128008, 128009]));
        let cfg = LlamaConfig::from_json_str(&v.to_string()).expect("array eos parses");
        assert_eq!(cfg.eos_token_id, 128001);
    }

    /// **M11.A.1 test** — empty array is a parse error (the field
    /// is structurally meaningless without at least one sentinel).
    #[test]
    fn eos_token_id_empty_array_rejected() {
        let v = base_config_value(json!([]));
        let err = LlamaConfig::from_json_str(&v.to_string()).expect_err("empty array must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("eos_token_id") && msg.contains("empty"),
            "error should mention field and emptiness, got: {msg}"
        );
    }

    /// **M11.A.1 test** — array whose first element is not an
    /// integer (e.g. accidentally `["128001"]`) is a parse error.
    #[test]
    fn eos_token_id_array_non_integer_first_rejected() {
        let v = base_config_value(json!(["not-a-number", 128001]));
        let err = LlamaConfig::from_json_str(&v.to_string())
            .expect_err("non-integer first element must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("eos_token_id"),
            "error should mention the field, got: {msg}"
        );
    }

    /// **M11.B test** — Phi-3.5 LongRope rope_scaling block parses
    /// into the new `RopeScaling::LongRope` variant. Reference
    /// shape from `models/phi-3.5-mini-instruct/config.json`:
    /// `type: "longrope"`, two factor arrays of length head_dim/2,
    /// and the original / max position embeddings at the top
    /// level of the config.
    #[test]
    fn rope_scaling_longrope_parses_into_variant() {
        let mut v = base_config_value(json!(32_000));
        // Top-level fields the LongRope branch needs.
        v["original_max_position_embeddings"] = json!(4096);
        v["max_position_embeddings"] = json!(131_072);
        // Factor vectors of length head_dim/2 = 32 (head_dim = 64
        // because hidden_size = 64 / num_heads = 4 in
        // base_config_value; LongRope only cares that
        // `factor.len() == head_dim/2`, the parser does not
        // validate the length here — that happens at
        // `compute_inv_freqs_longrope` call time).
        let short: Vec<f32> = (0..32).map(|i| 1.0 + i as f32 * 0.01).collect();
        let long: Vec<f32> = (0..32).map(|i| 2.0 + i as f32 * 0.05).collect();
        v["rope_scaling"] = json!({
            "type": "longrope",
            "short_factor": short,
            "long_factor": long
        });
        let cfg = LlamaConfig::from_json_str(&v.to_string()).expect("LongRope config must parse");
        match cfg.effective_rope_scaling() {
            Some(RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
                max_position_embeddings,
            }) => {
                assert_eq!(short_factor.len(), 32);
                assert_eq!(long_factor.len(), 32);
                assert!((short_factor[0] - 1.0).abs() < 1e-6);
                assert!((long_factor[0] - 2.0).abs() < 1e-6);
                assert_eq!(*original_max_position_embeddings, 4096);
                assert_eq!(*max_position_embeddings, 131_072);
            }
            other => panic!("expected RopeScaling::LongRope, got {other:?}"),
        }
    }

    /// **M11.C test** — minimal Gemma 2 config parses with the
    /// four new fields populated and `tie_word_embeddings`
    /// defaulted to `true` (Gemma 2 omits it from
    /// `config.json`). Mirrors the production
    /// `models/gemma-2-2b-it/config.json` shape: integer
    /// `query_pre_attn_scalar`, float caps, integer
    /// `sliding_window`, and missing `tie_word_embeddings`.
    #[test]
    fn gemma2_config_parses_with_softcaps_and_default_tie() {
        let v = json!({
            "architectures": ["Gemma2ForCausalLM"],
            "attn_logit_softcapping": 50.0,
            "bos_token_id": 2,
            "eos_token_id": [1, 107],
            "final_logit_softcapping": 30.0,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "max_position_embeddings": 8192,
            "model_type": "gemma2",
            "num_attention_heads": 8,
            "num_hidden_layers": 26,
            "num_key_value_heads": 4,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "sliding_window": 4096,
            "vocab_size": 256_000
            // tie_word_embeddings intentionally omitted.
        });
        let cfg = LlamaConfig::from_json_str(&v.to_string()).expect("Gemma 2 config must parse");
        assert_eq!(cfg.model_type.as_deref(), Some("gemma2"));
        assert!(cfg.tie_word_embeddings, "gemma2 default = true");
        assert!(
            cfg.attn_logit_softcapping
                .map(|x| (x - 50.0).abs() < 1e-6)
                .unwrap_or(false)
        );
        assert!(
            cfg.final_logit_softcapping
                .map(|x| (x - 30.0).abs() < 1e-6)
                .unwrap_or(false)
        );
        assert_eq!(cfg.sliding_window, Some(4096));
        assert!(
            cfg.query_pre_attn_scalar
                .map(|x| (x - 256.0).abs() < 1e-6)
                .unwrap_or(false)
        );
        assert_eq!(cfg.eos_token_id, 1);
        assert_eq!(cfg.head_dim, Some(256));
    }

    /// **M11.C test** — non-Gemma family with missing
    /// `tie_word_embeddings` still hard-errors. Guards the
    /// family-aware default from silently swallowing typos in
    /// future Llama-family configs.
    #[test]
    fn missing_tie_word_embeddings_errors_for_non_gemma() {
        let mut v = base_config_value(json!(2));
        v.as_object_mut().unwrap().remove("tie_word_embeddings");
        let err = LlamaConfig::from_json_str(&v.to_string())
            .expect_err("missing tie_word_embeddings on llama must fail");
        assert!(format!("{err}").contains("tie_word_embeddings"));
    }

    /// **M11.C test** — every Gemma 2 field defaults to `None`
    /// on a vanilla Llama config so existing fixtures stay
    /// byte-equivalent after the schema extension.
    #[test]
    fn llama_config_has_all_gemma2_fields_none() {
        let v = base_config_value(json!(2));
        let cfg = LlamaConfig::from_json_str(&v.to_string()).expect("plain Llama config parses");
        assert!(cfg.attn_logit_softcapping.is_none());
        assert!(cfg.final_logit_softcapping.is_none());
        assert!(cfg.sliding_window.is_none());
        assert!(cfg.query_pre_attn_scalar.is_none());
    }

    /// **M11.B test** — LongRope parser surfaces a clear error
    /// when `original_max_position_embeddings` is missing from
    /// the top level of the config (it lives outside the
    /// rope_scaling block per the Phi-3 schema).
    #[test]
    fn rope_scaling_longrope_missing_original_max_pos_fails() {
        let mut v = base_config_value(json!(32_000));
        v["max_position_embeddings"] = json!(131_072);
        v["rope_scaling"] = json!({
            "type": "longrope",
            "short_factor": [1.0, 1.0],
            "long_factor": [2.0, 2.0]
        });
        let err = LlamaConfig::from_json_str(&v.to_string())
            .expect_err("missing original_max_position_embeddings must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("original_max_position_embeddings"),
            "error should name the missing field, got: {msg}"
        );
    }
}
