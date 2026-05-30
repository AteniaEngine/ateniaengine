//! **MOE-FULL-2** — Mixture-of-Experts config fields (parse-only).
//!
//! This module parses the MoE-specific subset of a HuggingFace `config.json`
//! into a normalized [`MoeConfig`]. It is **parse-only and inert**:
//!
//! * It does NOT execute MoE, load MoE weights, or alter any dense model
//!   behaviour. No productive path consumes `MoeConfig` yet.
//! * It is fully decoupled from [`super::config::LlamaConfig`] — dense config
//!   parsing is untouched, so dense models behave exactly as before.
//! * The productive loader's MoE fail-loud guard (`LoaderError::MoeUnsupported`)
//!   is unchanged.
//!
//! Its only job is to give MOE-FULL-3+ a stable, family-agnostic way to read a
//! checkpoint's expert topology from config, normalizing the divergent field
//! names used across families:
//!
//! | Concept | Mixtral | Qwen-MoE | DeepSeek-MoE |
//! |---|---|---|---|
//! | routed experts | `num_local_experts` | `num_experts` | `n_routed_experts` |
//! | experts per token | `num_experts_per_tok` | `num_experts_per_tok` | `num_experts_per_tok` |
//! | shared experts | (none) | `shared_expert_intermediate_size` | `n_shared_experts` |
//! | renormalize top-k | (block always renorms) | `norm_topk_prob` | `norm_topk_prob` |
//! | expert FFN size | `intermediate_size` | `moe_intermediate_size` | `moe_intermediate_size` |
//!
//! `MoeConfig::is_moe()` is the single signal "this config describes a MoE
//! model". A dense config yields `is_moe() == false` with all fields `None`.

use serde_json::Value;

/// Normalized MoE configuration parsed from a `config.json`. Every routed/
/// expert field is optional and tolerant of absence — a dense model produces
/// an all-empty `MoeConfig` (`is_moe() == false`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoeConfig {
    /// Number of routed experts (normalized from `num_experts` /
    /// `num_local_experts` / `n_routed_experts`).
    pub num_experts: Option<usize>,
    /// Experts activated per token / top-k (`num_experts_per_tok` or
    /// `num_experts_per_token`).
    pub experts_per_token: Option<usize>,
    /// Number of shared experts, when expressed as a count
    /// (`n_shared_experts`, DeepSeek). `None` when not count-based.
    pub num_shared_experts: Option<usize>,
    /// Whether the model carries shared expert(s): true if `n_shared_experts`
    /// > 0 OR a `shared_expert_intermediate_size` is present (Qwen-MoE).
    pub has_shared_experts: bool,
    /// Whether the router renormalizes the selected top-k weights to sum 1
    /// (`norm_topk_prob`). `None` when the field is absent.
    pub norm_topk_prob: Option<bool>,
    /// Router auxiliary-loss coefficient (`router_aux_loss_coef`), if present.
    /// Informational only — not used at inference.
    pub router_aux_loss_coef: Option<f32>,
    /// Per-expert FFN intermediate size (normalized from
    /// `moe_intermediate_size` / `expert_intermediate_size`). `None` falls
    /// back to the dense `intermediate_size` at the caller's discretion.
    pub expert_intermediate_size: Option<usize>,
    /// Shared-expert FFN intermediate size (`shared_expert_intermediate_size`,
    /// Qwen-MoE), if present.
    pub shared_expert_intermediate_size: Option<usize>,
}

/// Read an optional `usize` from any of the given keys, first match wins.
fn first_usize(v: &Value, keys: &[&str]) -> Option<usize> {
    for k in keys {
        if let Some(n) = v.get(*k).and_then(Value::as_u64) {
            return Some(n as usize);
        }
    }
    None
}

/// Read an optional `f32` from a key.
fn opt_f32(v: &Value, key: &str) -> Option<f32> {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32)
}

/// Read an optional `bool` from a key.
fn opt_bool(v: &Value, key: &str) -> Option<bool> {
    v.get(key).and_then(Value::as_bool)
}

impl MoeConfig {
    /// Parse the MoE fields out of a `config.json` JSON string. Never fails on
    /// a dense config (returns an all-empty `MoeConfig`); only returns `Err`
    /// if the input is not valid JSON.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        let v: Value = serde_json::from_str(json)?;
        Ok(Self::from_value(&v))
    }

    /// Parse the MoE fields out of an already-deserialized config `Value`.
    /// Pure; tolerant of missing fields.
    pub fn from_value(v: &Value) -> Self {
        let num_experts = first_usize(v, &["num_experts", "num_local_experts", "n_routed_experts"]);
        let experts_per_token = first_usize(v, &["num_experts_per_tok", "num_experts_per_token"]);
        let num_shared_experts = first_usize(v, &["n_shared_experts"]);
        let shared_expert_intermediate_size = first_usize(v, &["shared_expert_intermediate_size"]);
        let expert_intermediate_size =
            first_usize(v, &["moe_intermediate_size", "expert_intermediate_size"]);

        // Shared experts present if a positive count OR a shared-expert FFN
        // size is declared (Qwen-MoE uses the size; DeepSeek uses the count).
        let has_shared_experts =
            num_shared_experts.is_some_and(|n| n > 0) || shared_expert_intermediate_size.is_some();

        MoeConfig {
            num_experts,
            experts_per_token,
            num_shared_experts,
            has_shared_experts,
            norm_topk_prob: opt_bool(v, "norm_topk_prob"),
            router_aux_loss_coef: opt_f32(v, "router_aux_loss_coef"),
            expert_intermediate_size,
            shared_expert_intermediate_size,
        }
    }

    /// Whether this config describes a Mixture-of-Experts model. The signal is
    /// the presence of a routed-expert count; a dense config has none.
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some()
    }

    /// The effective top-k, defaulting to `default` (e.g. when the field is
    /// absent), clamped to `[1, num_experts]` when the expert count is known.
    pub fn experts_per_token_or(&self, default: usize) -> usize {
        let k = self.experts_per_token.unwrap_or(default).max(1);
        match self.num_experts {
            Some(n) if n > 0 => k.min(n),
            _ => k,
        }
    }

    /// Whether the router renormalizes top-k weights. Mixtral's block always
    /// renormalizes regardless of the (often absent) `norm_topk_prob`, so the
    /// caller supplies the family default; an explicit field overrides it.
    pub fn renormalize_topk_or(&self, default: bool) -> bool {
        self.norm_topk_prob.unwrap_or(default)
    }
}

// ============================================================================
// Tests (synthetic config JSON only — no models, no loader, no execution)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- representative real config shapes (trimmed to relevant fields) ---

    fn mixtral_cfg() -> &'static str {
        r#"{
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "router_aux_loss_coef": 0.02
        }"#
    }

    fn qwen2_moe_cfg() -> &'static str {
        r#"{
            "architectures": ["Qwen2MoeForCausalLM"],
            "model_type": "qwen2_moe",
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 24,
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
            "norm_topk_prob": false,
            "router_aux_loss_coef": 0.001
        }"#
    }

    fn qwen3_moe_cfg() -> &'static str {
        r#"{
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "norm_topk_prob": true
        }"#
    }

    fn deepseek_moe_cfg() -> &'static str {
        r#"{
            "architectures": ["DeepseekV2ForCausalLM"],
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6,
            "n_shared_experts": 2,
            "moe_intermediate_size": 1408,
            "norm_topk_prob": false
        }"#
    }

    fn dense_qwen_cfg() -> &'static str {
        r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_hidden_layers": 28,
            "num_attention_heads": 12
        }"#
    }

    fn dense_mistral_cfg() -> &'static str {
        r#"{
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#
    }

    #[test]
    fn dense_config_remains_non_moe() {
        for cfg in [dense_qwen_cfg(), dense_mistral_cfg()] {
            let m = MoeConfig::from_json_str(cfg).unwrap();
            assert!(!m.is_moe(), "dense config must not be MoE");
            assert_eq!(m.num_experts, None);
            assert_eq!(m.experts_per_token, None);
            assert!(!m.has_shared_experts);
            assert_eq!(m, MoeConfig::default());
        }
    }

    #[test]
    fn mixtral_config_detects_moe() {
        let m = MoeConfig::from_json_str(mixtral_cfg()).unwrap();
        assert!(m.is_moe());
        assert_eq!(m.num_experts, Some(8)); // num_local_experts
        assert_eq!(m.experts_per_token, Some(2));
        assert!(!m.has_shared_experts); // Mixtral has no shared expert
        assert_eq!(m.norm_topk_prob, None); // absent -> family default applies
        assert_eq!(m.router_aux_loss_coef, Some(0.02));
        // Mixtral renormalizes by convention even though the field is absent.
        assert!(m.renormalize_topk_or(true));
    }

    #[test]
    fn qwen_moe_config_detects_moe() {
        let m = MoeConfig::from_json_str(qwen2_moe_cfg()).unwrap();
        assert!(m.is_moe());
        assert_eq!(m.num_experts, Some(60)); // num_experts
        assert_eq!(m.experts_per_token, Some(4));
        assert!(m.has_shared_experts); // via shared_expert_intermediate_size
        assert_eq!(m.shared_expert_intermediate_size, Some(5632));
        assert_eq!(m.expert_intermediate_size, Some(1408)); // moe_intermediate_size
        assert_eq!(m.norm_topk_prob, Some(false));
        assert!(!m.renormalize_topk_or(true)); // explicit false overrides default
    }

    #[test]
    fn qwen3_moe_config_detects_moe() {
        let m = MoeConfig::from_json_str(qwen3_moe_cfg()).unwrap();
        assert!(m.is_moe());
        assert_eq!(m.num_experts, Some(128));
        assert_eq!(m.experts_per_token, Some(8));
        assert!(!m.has_shared_experts); // Qwen3-MoE has no shared expert
        assert_eq!(m.norm_topk_prob, Some(true));
        assert_eq!(m.expert_intermediate_size, Some(768));
    }

    #[test]
    fn deepseek_moe_config_detects_shared_count() {
        let m = MoeConfig::from_json_str(deepseek_moe_cfg()).unwrap();
        assert!(m.is_moe());
        assert_eq!(m.num_experts, Some(64)); // n_routed_experts
        assert_eq!(m.experts_per_token, Some(6));
        assert_eq!(m.num_shared_experts, Some(2));
        assert!(m.has_shared_experts); // via positive count
        assert_eq!(m.norm_topk_prob, Some(false));
    }

    #[test]
    fn experts_per_token_aliases_normalize() {
        let a = MoeConfig::from_json_str(r#"{"num_experts":4,"num_experts_per_tok":2}"#).unwrap();
        let b = MoeConfig::from_json_str(r#"{"num_experts":4,"num_experts_per_token":2}"#).unwrap();
        assert_eq!(a.experts_per_token, Some(2));
        assert_eq!(b.experts_per_token, Some(2));
        assert_eq!(a.experts_per_token, b.experts_per_token);
    }

    #[test]
    fn num_experts_aliases_normalize() {
        let local = MoeConfig::from_json_str(r#"{"num_local_experts":8}"#).unwrap();
        let plain = MoeConfig::from_json_str(r#"{"num_experts":60}"#).unwrap();
        let routed = MoeConfig::from_json_str(r#"{"n_routed_experts":64}"#).unwrap();
        assert_eq!(local.num_experts, Some(8));
        assert_eq!(plain.num_experts, Some(60));
        assert_eq!(routed.num_experts, Some(64));
    }

    #[test]
    fn expert_intermediate_size_aliases_normalize() {
        let a = MoeConfig::from_json_str(r#"{"num_experts":4,"moe_intermediate_size":1408}"#).unwrap();
        let b =
            MoeConfig::from_json_str(r#"{"num_experts":4,"expert_intermediate_size":1408}"#).unwrap();
        assert_eq!(a.expert_intermediate_size, Some(1408));
        assert_eq!(b.expert_intermediate_size, Some(1408));
    }

    #[test]
    fn missing_fields_are_safe() {
        // Empty object: no MoE, no panic, all None.
        let m = MoeConfig::from_json_str("{}").unwrap();
        assert!(!m.is_moe());
        assert_eq!(m, MoeConfig::default());
        // experts_per_token_or falls back when absent.
        assert_eq!(m.experts_per_token_or(2), 2);
    }

    #[test]
    fn experts_per_token_clamps_to_expert_count() {
        let m = MoeConfig::from_json_str(r#"{"num_experts":4,"num_experts_per_tok":99}"#).unwrap();
        assert_eq!(m.experts_per_token_or(2), 4); // clamped to num_experts
    }

    #[test]
    fn invalid_json_errors() {
        assert!(MoeConfig::from_json_str("{ not json").is_err());
    }
}
