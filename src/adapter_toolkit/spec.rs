//! **Adapter Toolkit v2 — Part 3 (IR) + Part 5 (pattern catalog).**
//!
//! [`ResolvedAdapterSpec`] is the intermediate representation: the
//! validated, normalised form of an [`AdapterDsl`], with the family
//! mapped to a v1 [`ModelFamily`] and a base architecture string,
//! the feature flags collapsed into a [`FeatureSet`], and the
//! per-checkpoint overrides resolved into [`ResolvedOverrides`].
//!
//! **Part 5 — normalised pattern catalog.** The recurring
//! cross-family patterns the spec instructions call out — multi-EOS,
//! tied/untied embeddings, GQA expansion, fused QKV/MLP, RoPE
//! variants — are represented here as small declarative enums
//! ([`RopeKind`], [`AttentionKind`], [`KvHeadsResolved`]) plus the
//! boolean feature flags on [`FeatureSet`]. v1 already *executes*
//! these patterns (via `FamilyTensorSpec` / `ConfigPolicy` / the
//! family graph builders); this layer only *names* them so the
//! validator ([`super::validate`]) and introspection
//! ([`super::introspect`]) can reason about them declaratively.

use crate::model_adapters::ModelFamily;

use super::dsl::{AdapterDsl, ConfigSection, KvHeads, OverrideSection, TokenizerSection};
use super::ToolkitError;

/// RoPE variant — the normalised form of `config.rope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    /// Plain rotary embedding (Llama / Qwen / Mistral / Gemma /
    /// Falcon3).
    Standard,
    /// Phi-class LongRope (short/long factor interpolation).
    LongRope,
    /// Partial rotary (a fraction of head_dim is rotated).
    PartialRotary,
}

impl RopeKind {
    fn parse(s: &str) -> Result<Self, ToolkitError> {
        match s.to_ascii_lowercase().as_str() {
            "standard" | "default" | "rope" => Ok(RopeKind::Standard),
            "longrope" | "long_rope" => Ok(RopeKind::LongRope),
            "partial" | "partial_rotary" => Ok(RopeKind::PartialRotary),
            other => Err(ToolkitError::Resolution(format!(
                "unknown rope variant `{other}` (expected standard|longrope|partial)"
            ))),
        }
    }

    /// Human-readable label for introspection.
    pub fn label(self) -> &'static str {
        match self {
            RopeKind::Standard => "standard",
            RopeKind::LongRope => "longrope",
            RopeKind::PartialRotary => "partial_rotary",
        }
    }
}

/// Attention shape — the normalised form of `attention.type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    /// Multi-head attention (`num_kv_heads == num_heads`).
    Mha,
    /// Grouped-query attention (`1 < num_kv_heads < num_heads`).
    Gqa,
    /// Multi-query attention (`num_kv_heads == 1`).
    Mqa,
}

impl AttentionKind {
    fn parse(s: &str) -> Result<Self, ToolkitError> {
        match s.to_ascii_lowercase().as_str() {
            "mha" => Ok(AttentionKind::Mha),
            "gqa" => Ok(AttentionKind::Gqa),
            "mqa" => Ok(AttentionKind::Mqa),
            other => Err(ToolkitError::Resolution(format!(
                "unknown attention type `{other}` (expected mha|gqa|mqa)"
            ))),
        }
    }

    /// Human-readable label for introspection.
    pub fn label(self) -> &'static str {
        match self {
            AttentionKind::Mha => "mha",
            AttentionKind::Gqa => "gqa",
            AttentionKind::Mqa => "mqa",
        }
    }
}

/// Resolved KV-head declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvHeadsResolved {
    /// Defer to `config.json`'s `num_key_value_heads`.
    Auto,
    /// An explicit count from the DSL.
    Count(usize),
}

/// The normalised feature flags for a model. Part 5's "pattern
/// catalog" in struct form.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSet {
    /// RoPE variant.
    pub rope: RopeKind,
    /// Partial-rotary fraction, when `rope == PartialRotary`.
    pub partial_rotary_factor: Option<f32>,
    /// Attention shape.
    pub attention: AttentionKind,
    /// KV-head count (or `Auto`).
    pub kv_heads: KvHeadsResolved,
    /// Fused QKV projection weight present.
    pub fused_qkv: bool,
    /// Fused gate/up MLP projection weight present.
    pub fused_mlp: bool,
    /// Strategy name for splitting the fused QKV weight.
    pub split_strategy: Option<String>,
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self {
            rope: RopeKind::Standard,
            partial_rotary_factor: None,
            attention: AttentionKind::Mha,
            kv_heads: KvHeadsResolved::Auto,
            fused_qkv: false,
            fused_mlp: false,
            split_strategy: None,
        }
    }
}

/// Resolved tokenizer / generation overrides — both the base spec's
/// own tokenizer block and every per-checkpoint override.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedTokenizer {
    /// Explicit multi-EOS token id set.
    pub eos_tokens: Option<Vec<u32>>,
    /// Turn-terminator token strings.
    pub turn_terminators: Vec<String>,
}

impl ResolvedTokenizer {
    fn from_section(section: &TokenizerSection) -> Self {
        Self {
            eos_tokens: section.eos_tokens.clone(),
            turn_terminators: section.turn_terminators.clone().unwrap_or_default(),
        }
    }

    /// Layer `other` on top of `self`: any field `other` sets wins.
    fn layered_with(&self, other: &ResolvedTokenizer) -> Self {
        Self {
            eos_tokens: other.eos_tokens.clone().or_else(|| self.eos_tokens.clone()),
            turn_terminators: if other.turn_terminators.is_empty() {
                self.turn_terminators.clone()
            } else {
                other.turn_terminators.clone()
            },
        }
    }
}

/// A fully resolved per-checkpoint override.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOverride {
    /// The override label (the `overrides:` map key).
    pub label: String,
    /// Tokenizer settings after layering the override onto the base.
    pub tokenizer: ResolvedTokenizer,
}

/// The IR: a validated, normalised adapter spec.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAdapterSpec {
    /// v1 family the DSL `family` resolved to.
    pub family: ModelFamily,
    /// The HF architecture string used to resolve the v1 base
    /// adapter (explicit `architecture:` wins, else family default).
    pub architecture: &'static str,
    /// The HF `model_type` used as a secondary resolution key.
    pub model_type: &'static str,
    /// Normalised feature flags.
    pub features: FeatureSet,
    /// Base-spec tokenizer settings.
    pub tokenizer: ResolvedTokenizer,
    /// Per-checkpoint overrides, in DSL declaration order.
    pub overrides: Vec<ResolvedOverride>,
}

impl ResolvedAdapterSpec {
    /// Resolve and normalise an [`AdapterDsl`] into the IR. This is
    /// where the `family` string maps to a [`ModelFamily`] + base
    /// architecture, and where the feature enums are parsed. Pure
    /// structural validation (e.g. `gqa` requires `kv_heads`) is
    /// **not** done here — that is [`super::validate`]'s job — so
    /// the IR can be introspected even when it is inconsistent.
    pub fn resolve(dsl: &AdapterDsl) -> Result<Self, ToolkitError> {
        let (family, default_arch, default_model_type) = resolve_family(&dsl.family)?;

        // Explicit architecture/model_type override the family
        // defaults, but must still be one of the strings v1's
        // `resolve_adapter` recognises — the generator re-checks
        // this when it actually resolves the base adapter.
        let architecture = match dsl.architecture.as_deref() {
            Some(explicit) => intern_architecture(explicit)?,
            None => default_arch,
        };
        // An explicit `model_type` must be one v1 recognises. The
        // earlier implementation silently replaced an unrecognised
        // value with the family default — a silent error that hid
        // a real typo from the author. Now it fails loud.
        let model_type = match dsl.model_type.as_deref() {
            Some(explicit) => intern_model_type(explicit).ok_or_else(|| {
                ToolkitError::Resolution(format!(
                    "explicit model_type `{explicit}` is not one Atenia v1 recognises; \
                     supported: llama, qwen2, qwen3, gemma2, gemma3, phi3, mistral"
                ))
            })?,
            None => default_model_type,
        };

        let features = resolve_features(dsl)?;
        let tokenizer = dsl
            .tokenizer
            .as_ref()
            .map(ResolvedTokenizer::from_section)
            .unwrap_or_default();

        let mut overrides = Vec::with_capacity(dsl.overrides.len());
        for (label, section) in &dsl.overrides {
            overrides.push(resolve_override(label, section, &tokenizer));
        }

        Ok(Self {
            family,
            architecture,
            model_type,
            features,
            tokenizer,
            overrides,
        })
    }

    /// Look up a resolved override by label.
    pub fn override_for(&self, label: &str) -> Option<&ResolvedOverride> {
        self.overrides.iter().find(|o| o.label == label)
    }
}

fn resolve_override(
    label: &str,
    section: &OverrideSection,
    base_tokenizer: &ResolvedTokenizer,
) -> ResolvedOverride {
    let layered = match &section.tokenizer {
        Some(tok) => base_tokenizer.layered_with(&ResolvedTokenizer::from_section(tok)),
        None => base_tokenizer.clone(),
    };
    ResolvedOverride {
        label: label.to_string(),
        tokenizer: layered,
    }
}

fn resolve_features(dsl: &AdapterDsl) -> Result<FeatureSet, ToolkitError> {
    let mut f = FeatureSet::default();

    if let Some(cfg) = &dsl.config {
        let ConfigSection {
            rope,
            partial_rotary_factor,
            rope_theta: _,
        } = cfg;
        if let Some(r) = rope {
            f.rope = RopeKind::parse(r)?;
        }
        f.partial_rotary_factor = *partial_rotary_factor;
        // A declared partial_rotary_factor implies the partial
        // variant unless the author already said something else
        // explicitly — the validator catches the contradiction.
        if partial_rotary_factor.is_some() && rope.is_none() {
            f.rope = RopeKind::PartialRotary;
        }
    }

    if let Some(att) = &dsl.attention {
        if let Some(kind) = &att.kind {
            f.attention = AttentionKind::parse(kind)?;
        }
        if let Some(kv) = &att.kv_heads {
            f.kv_heads = match kv {
                KvHeads::Count(n) => KvHeadsResolved::Count(*n),
                KvHeads::Keyword(k) if k.eq_ignore_ascii_case("auto") => KvHeadsResolved::Auto,
                KvHeads::Keyword(other) => {
                    return Err(ToolkitError::Resolution(format!(
                        "kv_heads keyword `{other}` invalid (expected an integer or `auto`)"
                    )));
                }
            };
        }
    }

    if let Some(w) = &dsl.weights {
        f.fused_qkv = w.fused_qkv.unwrap_or(false);
        f.fused_mlp = w.fused_mlp.unwrap_or(false);
        f.split_strategy = w.split_strategy.clone();
    }

    Ok(f)
}

/// Map a DSL `family` string to `(ModelFamily, architecture,
/// model_type)`. The architecture/model_type strings are the v1
/// defaults `resolve_adapter` recognises.
fn resolve_family(
    family: &str,
) -> Result<(ModelFamily, &'static str, &'static str), ToolkitError> {
    match family.to_ascii_lowercase().as_str() {
        "llama" => Ok((ModelFamily::Llama, "LlamaForCausalLM", "llama")),
        "qwen" | "qwen2" => Ok((ModelFamily::Qwen2, "Qwen2ForCausalLM", "qwen2")),
        "qwen3" => Ok((ModelFamily::Qwen3, "Qwen3ForCausalLM", "qwen3")),
        "gemma" | "gemma2" => Ok((ModelFamily::Gemma2, "Gemma2ForCausalLM", "gemma2")),
        "gemma3" => Ok((ModelFamily::Gemma3, "Gemma3ForCausalLM", "gemma3")),
        "phi" | "phi3" => Ok((ModelFamily::Phi3, "Phi3ForCausalLM", "phi3")),
        "mistral" => Ok((ModelFamily::Mistral, "MistralForCausalLM", "mistral")),
        other => Err(ToolkitError::Resolution(format!(
            "unknown family `{other}` — Adapter Toolkit v2 supports \
             llama, qwen/qwen2, qwen3, gemma/gemma2, gemma3, phi/phi3, mistral"
        ))),
    }
}

/// Intern an explicit architecture string to a `&'static str` v1
/// recognises. An unknown architecture is rejected here so the
/// failure is a clear toolkit error rather than a later silent
/// fallback to the Llama adapter.
fn intern_architecture(arch: &str) -> Result<&'static str, ToolkitError> {
    match arch {
        "LlamaForCausalLM" => Ok("LlamaForCausalLM"),
        "Qwen2ForCausalLM" => Ok("Qwen2ForCausalLM"),
        "Qwen3ForCausalLM" => Ok("Qwen3ForCausalLM"),
        "Gemma2ForCausalLM" => Ok("Gemma2ForCausalLM"),
        "Gemma3ForCausalLM" => Ok("Gemma3ForCausalLM"),
        "Phi3ForCausalLM" => Ok("Phi3ForCausalLM"),
        "MistralForCausalLM" => Ok("MistralForCausalLM"),
        other => Err(ToolkitError::Resolution(format!(
            "explicit architecture `{other}` is not one Atenia v1 resolves; \
             supported: LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, \
             Gemma2ForCausalLM, Gemma3ForCausalLM, Phi3ForCausalLM, MistralForCausalLM"
        ))),
    }
}

fn intern_model_type(model_type: &str) -> Option<&'static str> {
    match model_type {
        "llama" => Some("llama"),
        "qwen2" => Some("qwen2"),
        "qwen3" => Some("qwen3"),
        "gemma2" => Some("gemma2"),
        "gemma3" => Some("gemma3"),
        "phi3" => Some("phi3"),
        "mistral" => Some("mistral"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dsl(text: &str) -> AdapterDsl {
        AdapterDsl::from_str(text, true).expect("dsl parses")
    }

    #[test]
    fn family_llama_resolves_to_v1_family() {
        let spec = ResolvedAdapterSpec::resolve(&dsl("family: llama\n")).expect("resolves");
        assert_eq!(spec.family, ModelFamily::Llama);
        assert_eq!(spec.architecture, "LlamaForCausalLM");
        assert_eq!(spec.model_type, "llama");
    }

    #[test]
    fn qwen_alias_resolves_to_qwen2() {
        let spec = ResolvedAdapterSpec::resolve(&dsl("family: qwen\n")).expect("resolves");
        assert_eq!(spec.family, ModelFamily::Qwen2);
        assert_eq!(spec.architecture, "Qwen2ForCausalLM");
    }

    #[test]
    fn qwen3_is_distinct_from_qwen2() {
        let spec = ResolvedAdapterSpec::resolve(&dsl("family: qwen3\n")).expect("resolves");
        assert_eq!(spec.family, ModelFamily::Qwen3);
        assert_eq!(spec.architecture, "Qwen3ForCausalLM");
    }

    #[test]
    fn unknown_family_is_typed_error() {
        let err = ResolvedAdapterSpec::resolve(&dsl("family: falcon\n"));
        assert!(matches!(err, Err(ToolkitError::Resolution(_))));
    }

    #[test]
    fn explicit_architecture_overrides_family_default() {
        let spec = ResolvedAdapterSpec::resolve(&dsl(
            "family: qwen\narchitecture: Qwen2ForCausalLM\n",
        ))
        .expect("resolves");
        assert_eq!(spec.architecture, "Qwen2ForCausalLM");
    }

    #[test]
    fn explicit_unknown_architecture_is_rejected() {
        let err = ResolvedAdapterSpec::resolve(&dsl(
            "family: llama\narchitecture: FalconForCausalLM\n",
        ));
        assert!(matches!(err, Err(ToolkitError::Resolution(_))));
    }

    #[test]
    fn explicit_unknown_model_type_is_rejected_not_silently_swapped() {
        // Regression: an unrecognised explicit model_type used to
        // be silently replaced with the family default. It must
        // now fail loud so a typo is visible.
        let err = ResolvedAdapterSpec::resolve(&dsl(
            "family: qwen\nmodel_type: qwen_typo\n",
        ));
        assert!(matches!(err, Err(ToolkitError::Resolution(_))));
    }

    #[test]
    fn explicit_known_model_type_is_honoured() {
        let spec = ResolvedAdapterSpec::resolve(&dsl(
            "family: qwen\nmodel_type: qwen2\n",
        ))
        .expect("resolves");
        assert_eq!(spec.model_type, "qwen2");
    }

    #[test]
    fn rope_and_attention_features_normalise() {
        let spec = ResolvedAdapterSpec::resolve(&dsl(
            "family: phi\nconfig:\n  rope: longrope\nattention:\n  type: gqa\n  kv_heads: 8\n",
        ))
        .expect("resolves");
        assert_eq!(spec.features.rope, RopeKind::LongRope);
        assert_eq!(spec.features.attention, AttentionKind::Gqa);
        assert_eq!(spec.features.kv_heads, KvHeadsResolved::Count(8));
    }

    #[test]
    fn partial_rotary_factor_implies_partial_rope() {
        let spec = ResolvedAdapterSpec::resolve(&dsl(
            "family: phi\nconfig:\n  partial_rotary_factor: 0.75\n",
        ))
        .expect("resolves");
        assert_eq!(spec.features.rope, RopeKind::PartialRotary);
        assert_eq!(spec.features.partial_rotary_factor, Some(0.75));
    }

    #[test]
    fn kv_heads_auto_keyword_resolves() {
        let spec = ResolvedAdapterSpec::resolve(&dsl(
            "family: qwen\nattention:\n  type: gqa\n  kv_heads: auto\n",
        ))
        .expect("resolves");
        assert_eq!(spec.features.kv_heads, KvHeadsResolved::Auto);
    }

    #[test]
    fn override_layers_eos_tokens_onto_base() {
        let spec = ResolvedAdapterSpec::resolve(&dsl(
            "family: qwen\n\
             tokenizer:\n  eos_tokens: [151643]\n\
             overrides:\n  deepseek-distill:\n    tokenizer:\n      eos_tokens: [1, 106]\n",
        ))
        .expect("resolves");
        assert_eq!(spec.tokenizer.eos_tokens, Some(vec![151643]));
        let ov = spec.override_for("deepseek-distill").expect("override");
        assert_eq!(ov.tokenizer.eos_tokens, Some(vec![1, 106]));
    }

    #[test]
    fn override_without_tokenizer_inherits_base() {
        let spec = ResolvedAdapterSpec::resolve(&dsl(
            "family: llama\n\
             tokenizer:\n  eos_tokens: [2]\n\
             overrides:\n  variant-a:\n    config:\n      rope: standard\n",
        ))
        .expect("resolves");
        let ov = spec.override_for("variant-a").expect("override");
        assert_eq!(ov.tokenizer.eos_tokens, Some(vec![2]));
    }
}
