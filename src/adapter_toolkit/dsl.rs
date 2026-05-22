//! **Adapter Toolkit v2 — Part 2: the declarative DSL.**
//!
//! Pure serde schema for the adapter spec file. One struct,
//! [`AdapterDsl`], covers all three documented authoring levels
//! (simple / intermediate / advanced) — every section beyond
//! `family` is optional, so a one-line `family: llama` file and a
//! fully-specified advanced file deserialize into the same type.
//!
//! This module is **data only**: no v1 type is referenced, no
//! adapter is constructed. [`super::spec`] turns an `AdapterDsl`
//! into the resolved IR; [`super::generator`] turns that into a
//! live adapter. Parsing both `.yaml`/`.yml` (serde_yaml) and
//! `.json` (serde_json) lands here via [`AdapterDsl::from_str`] /
//! [`AdapterDsl::from_file`].

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::ToolkitError;

/// The complete declarative adapter spec. `family` is the only
/// required field; everything else is optional and defaults to
/// "unspecified" so the three authoring levels share one type.
///
/// `deny_unknown_fields` makes a misspelled key a hard parse error
/// rather than a silently-ignored field — the toolkit's first line
/// of "validación automática".
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterDsl {
    /// Model family selector. Accepted (case-insensitive):
    /// `llama`, `qwen` / `qwen2`, `qwen3`, `gemma` / `gemma2`,
    /// `gemma3`, `phi` / `phi3`, `mistral`. Resolution lives in
    /// [`super::spec`].
    pub family: String,

    /// Explicit HF `architectures[0]` string (e.g.
    /// `Qwen2ForCausalLM`). When present it overrides the
    /// family-derived default during adapter resolution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,

    /// Explicit HF `model_type` (e.g. `qwen2`). Informational +
    /// used as a secondary resolution key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,

    /// Weight format: `safetensors` or `gguf`. Informational —
    /// the loader detects the real format from the model dir.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Quantisation tag (e.g. `Q4_K_M`). Informational only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,

    /// Advanced: config-level declarations (rope variant, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<ConfigSection>,

    /// Advanced: weight-layout declarations (fused QKV / MLP).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<WeightsSection>,

    /// Advanced: attention-shape declarations (MHA / GQA / MQA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention: Option<AttentionSection>,

    /// Tokenizer declarations (EOS set, turn terminators).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerSection>,

    /// Per-checkpoint overrides, keyed by an opaque label
    /// (e.g. `deepseek-distill`). Each entry layers on top of the
    /// base spec.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub overrides: BTreeMap<String, OverrideSection>,
}

/// Config-level declarations.
///
/// **Semantics — declarative, not authoritative.** Every field here
/// is an *expected constraint*: the toolkit parses it, normalises
/// it into the [`super::spec::FeatureSet`], and uses it for
/// validation and introspection. It does **not** mutate the
/// model's `LlamaConfig` — `config.json` / GGUF metadata stay the
/// single source of truth for the runtime. A field left unset
/// simply means "no declared expectation; trust the model".
#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigSection {
    /// RoPE variant: `standard`, `longrope`, `partial` /
    /// `partial_rotary`. Absent ⇒ `standard`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope: Option<String>,

    /// Partial-rotary fraction (Phi-class). Declaring this without
    /// a `rope` of `partial` is a validation error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f32>,

    /// RoPE base frequency. Informational/declarative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f32>,
}

/// Weight-layout declarations.
///
/// **Semantics — declarative, not authoritative** (see
/// [`ConfigSection`]). These fields describe the weight layout the
/// model is *expected* to have; the v1 family adapter owns the
/// actual fused-weight handling. Declaring `fused_qkv: true` on a
/// family whose v1 builder has no fused path is a validation
/// *warning*, not a behaviour change.
#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WeightsSection {
    /// Fused QKV projection (Phi-class). Requires `split_strategy`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_qkv: Option<bool>,

    /// Fused gate/up MLP projection (Phi-class).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_mlp: Option<bool>,

    /// Strategy name for splitting a fused QKV weight, e.g.
    /// `phi_qkv`. Required whenever `fused_qkv: true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_strategy: Option<String>,
}

/// Attention-shape declarations.
///
/// **Semantics — declarative, not authoritative** (see
/// [`ConfigSection`]). `kv_heads` is validated, not injected: the
/// real head counts come from `config.json` / GGUF metadata.
#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionSection {
    /// `mha`, `gqa`, or `mqa`. Absent ⇒ `mha`.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,

    /// KV-head count. A bare integer, or the keyword `auto` to
    /// defer to `config.json`'s `num_key_value_heads`. `gqa`
    /// without `kv_heads` is a validation error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_heads: Option<KvHeads>,
}

/// KV-head declaration: an explicit count or the `auto` keyword.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum KvHeads {
    /// An explicit integer count.
    Count(usize),
    /// A keyword — only `auto` is accepted (validated in
    /// [`super::spec`]).
    Keyword(String),
}

/// Tokenizer declarations.
#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSection {
    /// Explicit EOS / stop token id set. Maps onto v1's multi-EOS
    /// (`GenerationConfig.eos_token_ids`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_tokens: Option<Vec<u32>>,

    /// Turn-terminator token strings (e.g. `<|im_end|>`). Resolved
    /// to ids by vocab lookup in the generation pipeline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_terminators: Option<Vec<String>>,
}

/// A per-checkpoint override block. Only the sections that a
/// specific checkpoint needs to change are present; each layers
/// onto the base spec.
#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OverrideSection {
    /// Tokenizer overrides for this checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerSection>,

    /// Config overrides for this checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<ConfigSection>,
}

impl AdapterDsl {
    /// Parse a DSL document from a string. `is_yaml` selects the
    /// backend: `true` ⇒ serde_yaml, `false` ⇒ serde_json.
    ///
    /// TODO(adapter-toolkit): `serde_yaml` 0.9 is deprecated
    /// upstream (see the Cargo.toml note). Migration is isolated to
    /// the single `serde_yaml::from_str` call below plus the
    /// `serde_yaml::to_string` call in [`AdapterDsl::to_yaml`]. The
    /// JSON backend is fully maintained and behaviour-equivalent.
    pub fn from_str(text: &str, is_yaml: bool) -> Result<Self, ToolkitError> {
        if is_yaml {
            serde_yaml::from_str(text).map_err(|e| ToolkitError::Parse(e.to_string()))
        } else {
            serde_json::from_str(text).map_err(|e| ToolkitError::Parse(e.to_string()))
        }
    }

    /// Parse a DSL document from a file, choosing the backend from
    /// the extension (`.yaml`/`.yml` ⇒ YAML, `.json` ⇒ JSON). An
    /// unrecognised extension is a typed error.
    pub fn from_file(path: &Path) -> Result<Self, ToolkitError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        let is_yaml = match ext.as_str() {
            "yaml" | "yml" => true,
            "json" => false,
            other => {
                return Err(ToolkitError::UnsupportedExtension(other.to_string()));
            }
        };
        let text = std::fs::read_to_string(path)
            .map_err(|e| ToolkitError::Io(format!("{}: {e}", path.display())))?;
        Self::from_str(&text, is_yaml)
    }

    /// Serialize this DSL back to YAML — used by `atenia inspect`
    /// to emit a generated spec.
    pub fn to_yaml(&self) -> Result<String, ToolkitError> {
        serde_yaml::to_string(self).map_err(|e| ToolkitError::Parse(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_level_one_line_parses() {
        let dsl = AdapterDsl::from_str("family: llama\n", true).expect("parses");
        assert_eq!(dsl.family, "llama");
        assert!(dsl.config.is_none());
        assert!(dsl.overrides.is_empty());
    }

    #[test]
    fn intermediate_level_parses() {
        let text = "family: qwen\narchitecture: Qwen2ForCausalLM\ntokenizer:\n  eos_tokens: [1, 106]\n";
        let dsl = AdapterDsl::from_str(text, true).expect("parses");
        assert_eq!(dsl.family, "qwen");
        assert_eq!(dsl.architecture.as_deref(), Some("Qwen2ForCausalLM"));
        assert_eq!(
            dsl.tokenizer.unwrap().eos_tokens.unwrap(),
            vec![1, 106]
        );
    }

    #[test]
    fn advanced_level_parses_all_sections() {
        let text = "\
family: phi
architecture: Phi3ForCausalLM
config:
  rope: longrope
  partial_rotary_factor: 0.75
weights:
  fused_qkv: true
  split_strategy: phi_qkv
attention:
  type: gqa
  kv_heads: auto
tokenizer:
  turn_terminators:
    - \"<|end|>\"
overrides:
  deepseek-distill:
    tokenizer:
      eos_tokens: [1, 106]
";
        let dsl = AdapterDsl::from_str(text, true).expect("parses");
        assert_eq!(dsl.family, "phi");
        let cfg = dsl.config.unwrap();
        assert_eq!(cfg.rope.as_deref(), Some("longrope"));
        assert_eq!(cfg.partial_rotary_factor, Some(0.75));
        let w = dsl.weights.unwrap();
        assert_eq!(w.fused_qkv, Some(true));
        assert_eq!(w.split_strategy.as_deref(), Some("phi_qkv"));
        let a = dsl.attention.unwrap();
        assert_eq!(a.kind.as_deref(), Some("gqa"));
        assert_eq!(a.kv_heads, Some(KvHeads::Keyword("auto".to_string())));
        assert_eq!(
            dsl.tokenizer.unwrap().turn_terminators.unwrap(),
            vec!["<|end|>".to_string()]
        );
        let ov = dsl.overrides.get("deepseek-distill").expect("override");
        assert_eq!(
            ov.tokenizer.as_ref().unwrap().eos_tokens.as_ref().unwrap(),
            &vec![1, 106]
        );
    }

    #[test]
    fn kv_heads_accepts_integer() {
        let dsl = AdapterDsl::from_str(
            "family: qwen\nattention:\n  type: gqa\n  kv_heads: 8\n",
            true,
        )
        .expect("parses");
        assert_eq!(
            dsl.attention.unwrap().kv_heads,
            Some(KvHeads::Count(8))
        );
    }

    #[test]
    fn json_form_parses_identically() {
        let json = r#"{"family":"llama","tokenizer":{"eos_tokens":[2]}}"#;
        let dsl = AdapterDsl::from_str(json, false).expect("json parses");
        assert_eq!(dsl.family, "llama");
        assert_eq!(dsl.tokenizer.unwrap().eos_tokens.unwrap(), vec![2]);
    }

    #[test]
    fn unknown_field_is_hard_error() {
        let err = AdapterDsl::from_str("family: llama\nbogus_key: 1\n", true);
        assert!(err.is_err(), "deny_unknown_fields must reject typos");
    }

    #[test]
    fn missing_family_is_hard_error() {
        assert!(AdapterDsl::from_str("architecture: LlamaForCausalLM\n", true).is_err());
    }

    #[test]
    fn yaml_round_trips_through_serialization() {
        let text = "family: gemma\nconfig:\n  rope: standard\n";
        let dsl = AdapterDsl::from_str(text, true).expect("parses");
        let yaml = dsl.to_yaml().expect("serializes");
        let reparsed = AdapterDsl::from_str(&yaml, true).expect("reparses");
        assert_eq!(dsl, reparsed);
    }
}
