//! **MOE-2** — MoE tensor-name detection + role classification (pure).
//!
//! This module recognises Mixture-of-Experts tensor names in a checkpoint
//! and classifies each into a [`TensorRole`], extracting layer / expert
//! ids where present. It is **detection only** — no router, no top-k, no
//! dispatch, no execution, no graph/runtime touch.
//!
//! Its purpose is to let the loader **fail loud** on a MoE checkpoint
//! (whose experts cannot be executed yet) instead of silently skipping the
//! expert tensors and loading a broken, half-dense model. See
//! [`detect_moe`] and `LoaderError::MoeUnsupported`.
//!
//! ## Recognised patterns
//!
//! Mixtral-style:
//! ```text
//! model.layers.{L}.block_sparse_moe.gate.weight          (router)
//! model.layers.{L}.block_sparse_moe.experts.{E}.w1.weight (expert gate)
//! model.layers.{L}.block_sparse_moe.experts.{E}.w2.weight (expert down)
//! model.layers.{L}.block_sparse_moe.experts.{E}.w3.weight (expert up)
//! ```
//!
//! Qwen-MoE / DeepSeek-MoE-style:
//! ```text
//! model.layers.{L}.mlp.gate.weight                          (router)
//! model.layers.{L}.mlp.experts.{E}.gate_proj.weight         (expert gate)
//! model.layers.{L}.mlp.experts.{E}.up_proj.weight           (expert up)
//! model.layers.{L}.mlp.experts.{E}.down_proj.weight         (expert down)
//! model.layers.{L}.mlp.shared_expert.*  /  shared_experts.* (shared expert)
//! ```
//!
//! The **fail-loud trigger** is the presence of *expert* tensors — names
//! containing `block_sparse_moe`, `.experts.`, or `shared_expert(s)`.
//! These substrings never appear in dense checkpoints (dense SwiGLU uses
//! `mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj`, no `.experts.`), so
//! dense Llama / Qwen / Mistral / Phi / Gemma / Falcon and dense
//! DeepSeek-R1 *distill* models do not trigger detection.

/// The role a tensor plays, including MoE-specific roles. Dense roles are
/// included so the same classifier can describe any linear weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    MlpGate,
    MlpUp,
    MlpDown,
    MoeRouter,
    MoeExpertGate,
    MoeExpertUp,
    MoeExpertDown,
    MoeSharedExpert,
    /// **MOE-15** — packed/fused gate+up tensor for ALL experts of a layer
    /// (`mlp.experts.gate_up_proj`, 3-D `[num_experts, 2*d_ff, d_model]`).
    MoePackedGateUp,
    /// **MOE-15** — packed/fused down tensor for ALL experts of a layer
    /// (`mlp.experts.down_proj`, 3-D `[num_experts, d_model, d_ff]`).
    MoePackedDown,
    Unknown,
}

impl TensorRole {
    /// Whether this role is MoE-specific (router / expert / shared expert).
    pub fn is_moe(&self) -> bool {
        matches!(
            self,
            TensorRole::MoeRouter
                | TensorRole::MoeExpertGate
                | TensorRole::MoeExpertUp
                | TensorRole::MoeExpertDown
                | TensorRole::MoeSharedExpert
                | TensorRole::MoePackedGateUp
                | TensorRole::MoePackedDown
        )
    }

    /// Whether this role is an actual *expert* weight (the fail-loud
    /// trigger), as opposed to the router alone.
    pub fn is_moe_expert(&self) -> bool {
        matches!(
            self,
            TensorRole::MoeExpertGate
                | TensorRole::MoeExpertUp
                | TensorRole::MoeExpertDown
                | TensorRole::MoeSharedExpert
                | TensorRole::MoePackedGateUp
                | TensorRole::MoePackedDown
        )
    }

    /// Whether this role is a packed/fused expert tensor (MOE-15).
    pub fn is_moe_packed(&self) -> bool {
        matches!(self, TensorRole::MoePackedGateUp | TensorRole::MoePackedDown)
    }
}

/// Classification of one tensor name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorNameInfo {
    pub role: TensorRole,
    pub layer_id: Option<usize>,
    pub expert_id: Option<usize>,
}

/// Parse the numeric id that follows a `marker` segment in `name`, e.g.
/// `extract_id_after(name, "layers.")` → the layer index. Returns `None`
/// if the marker is absent or not followed by digits.
fn extract_id_after(name: &str, marker: &str) -> Option<usize> {
    let start = name.find(marker)? + marker.len();
    let rest = &name[start..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<usize>().ok()
    }
}

/// Whether a tensor name belongs to a MoE *expert* (the fail-loud
/// trigger). Router-only names are intentionally NOT included here.
pub fn is_moe_expert_tensor(name: &str) -> bool {
    name.contains("block_sparse_moe.experts.")
        || name.contains(".mlp.experts.")
        || name.contains(".experts.")
        || name.contains("shared_expert")
}

/// Whether a tensor name looks like a MoE router gate.
///
/// Distinguishes the router (`...block_sparse_moe.gate.weight` /
/// `...mlp.gate.weight`) from a dense SwiGLU gate projection
/// (`...mlp.gate_proj.weight`), which is NOT a router.
pub fn is_moe_router_tensor(name: &str) -> bool {
    name.contains("block_sparse_moe.gate.")
        || (name.contains(".mlp.gate.") && !name.contains("gate_proj"))
}

/// Classify a single tensor name into a [`TensorNameInfo`].
pub fn classify_tensor_name(name: &str) -> TensorNameInfo {
    let layer_id = extract_id_after(name, ".layers.");
    let expert_id = extract_id_after(name, ".experts.");

    let role = if name.contains("shared_expert") {
        TensorRole::MoeSharedExpert
    } else if name.contains("experts.gate_up_proj") {
        // MOE-15: packed/fused gate+up for all experts (no per-expert id).
        TensorRole::MoePackedGateUp
    } else if name.contains("experts.") && expert_id.is_none() && name.contains("down_proj") {
        // MOE-15: packed/fused down for all experts (`experts.down_proj`,
        // no per-expert id — classic down carries an id and is handled below).
        TensorRole::MoePackedDown
    } else if name.contains(".experts.") || name.contains("block_sparse_moe.experts.") {
        // Expert weight: classify the inner projection.
        if name.contains(".w1.") || name.contains(".gate_proj.") {
            TensorRole::MoeExpertGate
        } else if name.contains(".w3.") || name.contains(".up_proj.") {
            TensorRole::MoeExpertUp
        } else if name.contains(".w2.") || name.contains(".down_proj.") {
            TensorRole::MoeExpertDown
        } else {
            TensorRole::Unknown
        }
    } else if is_moe_router_tensor(name) {
        TensorRole::MoeRouter
    } else if name.contains("q_proj") {
        TensorRole::AttentionQ
    } else if name.contains("k_proj") {
        TensorRole::AttentionK
    } else if name.contains("v_proj") {
        TensorRole::AttentionV
    } else if name.contains("o_proj") {
        TensorRole::AttentionO
    } else if name.contains("gate_proj") {
        TensorRole::MlpGate
    } else if name.contains("up_proj") {
        TensorRole::MlpUp
    } else if name.contains("down_proj") {
        TensorRole::MlpDown
    } else {
        TensorRole::Unknown
    };

    TensorNameInfo {
        role,
        layer_id,
        expert_id,
    }
}

/// Aggregate detection result over a checkpoint's tensor names.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoeDetection {
    /// True iff at least one MoE *expert* tensor was found.
    pub is_moe: bool,
    /// Count of expert-weight tensors (gate/up/down across all experts).
    pub expert_tensor_count: usize,
    /// Count of router-gate tensors.
    pub router_tensor_count: usize,
    /// Count of shared-expert tensors.
    pub shared_expert_tensor_count: usize,
    /// Highest expert id observed (`Some(max)`), if any expert ids parsed.
    pub max_expert_id: Option<usize>,
}

impl MoeDetection {
    /// Number of distinct experts implied by `max_expert_id` (`max + 1`),
    /// or `None` if no expert ids were parsed.
    pub fn implied_expert_count(&self) -> Option<usize> {
        self.max_expert_id.map(|m| m + 1)
    }
}

/// Scan a checkpoint's tensor names and report MoE detection. Pure; does
/// not load anything.
pub fn detect_moe<'a, I>(names: I) -> MoeDetection
where
    I: IntoIterator<Item = &'a str>,
{
    let mut det = MoeDetection::default();
    for name in names {
        let info = classify_tensor_name(name);
        match info.role {
            TensorRole::MoeRouter => det.router_tensor_count += 1,
            TensorRole::MoeSharedExpert => {
                det.shared_expert_tensor_count += 1;
                det.is_moe = true;
            }
            TensorRole::MoeExpertGate
            | TensorRole::MoeExpertUp
            | TensorRole::MoeExpertDown
            | TensorRole::MoePackedGateUp
            | TensorRole::MoePackedDown => {
                det.expert_tensor_count += 1;
                det.is_moe = true;
            }
            _ => {}
        }
        if let Some(eid) = info.expert_id {
            det.max_expert_id = Some(det.max_expert_id.map_or(eid, |m| m.max(eid)));
        }
    }
    det
}

/// Build the human-facing fail-loud message for a detected MoE checkpoint.
/// Used by the loader to populate `LoaderError::MoeUnsupported`.
pub fn unsupported_message(det: &MoeDetection) -> String {
    let experts = det
        .implied_expert_count()
        .map(|n| n.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "MoE checkpoint detected, but MoE execution is not implemented yet \
         (experts={experts}, expert_tensors={}, router_tensors={}, shared_expert_tensors={}). \
         Atenia can load and run dense models only; loading this checkpoint as dense would \
         silently drop the expert weights and produce a broken model. See \
         docs/MOE_CERTIFICATION_SUBSTRATE.md and docs/HANDOFF_MOE_2.md.",
        det.expert_tensor_count, det.router_tensor_count, det.shared_expert_tensor_count
    )
}

// ============================================================================
// Tests (synthetic names only — no model load)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_mixtral_expert_tensor_names() {
        let n = "model.layers.3.block_sparse_moe.experts.5.w1.weight";
        let info = classify_tensor_name(n);
        assert_eq!(info.role, TensorRole::MoeExpertGate);
        assert_eq!(info.layer_id, Some(3));
        assert_eq!(info.expert_id, Some(5));
        assert!(is_moe_expert_tensor(n));
        // w2 = down, w3 = up.
        assert_eq!(
            classify_tensor_name("model.layers.0.block_sparse_moe.experts.0.w2.weight").role,
            TensorRole::MoeExpertDown
        );
        assert_eq!(
            classify_tensor_name("model.layers.0.block_sparse_moe.experts.0.w3.weight").role,
            TensorRole::MoeExpertUp
        );
    }

    #[test]
    fn detects_moe_router_tensor_names() {
        assert_eq!(
            classify_tensor_name("model.layers.2.block_sparse_moe.gate.weight").role,
            TensorRole::MoeRouter
        );
        assert_eq!(
            classify_tensor_name("model.layers.2.mlp.gate.weight").role,
            TensorRole::MoeRouter
        );
        assert!(is_moe_router_tensor("model.layers.2.mlp.gate.weight"));
    }

    #[test]
    fn extracts_layer_and_expert_ids() {
        let info = classify_tensor_name("model.layers.11.mlp.experts.42.up_proj.weight");
        assert_eq!(info.layer_id, Some(11));
        assert_eq!(info.expert_id, Some(42));
        assert_eq!(info.role, TensorRole::MoeExpertUp);
    }

    #[test]
    fn dense_tensor_names_still_map_normally() {
        assert_eq!(
            classify_tensor_name("model.layers.0.self_attn.q_proj.weight").role,
            TensorRole::AttentionQ
        );
        assert_eq!(
            classify_tensor_name("model.layers.0.mlp.gate_proj.weight").role,
            TensorRole::MlpGate
        );
        assert_eq!(
            classify_tensor_name("model.layers.0.mlp.up_proj.weight").role,
            TensorRole::MlpUp
        );
        assert_eq!(
            classify_tensor_name("model.layers.0.mlp.down_proj.weight").role,
            TensorRole::MlpDown
        );
        // Dense gate_proj must NOT be misread as a router.
        assert!(!is_moe_router_tensor("model.layers.0.mlp.gate_proj.weight"));
        assert!(!is_moe_expert_tensor("model.layers.0.mlp.gate_proj.weight"));
    }

    #[test]
    fn shared_expert_name_triggers_moe_detection() {
        let n = "model.layers.0.mlp.shared_expert.gate_proj.weight";
        assert_eq!(classify_tensor_name(n).role, TensorRole::MoeSharedExpert);
        let det = detect_moe([n]);
        assert!(det.is_moe);
        assert_eq!(det.shared_expert_tensor_count, 1);
    }

    #[test]
    fn moe_checkpoint_detected() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
            "model.layers.0.block_sparse_moe.experts.7.w1.weight",
        ];
        let det = detect_moe(names);
        assert!(det.is_moe);
        assert_eq!(det.expert_tensor_count, 4);
        assert_eq!(det.router_tensor_count, 1);
        assert_eq!(det.max_expert_id, Some(7));
        assert_eq!(det.implied_expert_count(), Some(8));
    }

    #[test]
    fn deepseek_distill_dense_does_not_trigger_moe() {
        // DeepSeek-R1-Distill is a DENSE Llama/Qwen derivative — standard
        // mlp.*_proj names, NO experts.
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ];
        let det = detect_moe(names);
        assert!(!det.is_moe);
        assert_eq!(det.expert_tensor_count, 0);
        assert_eq!(det.router_tensor_count, 0);
    }

    #[test]
    fn qwen_dense_does_not_trigger_moe() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "lm_head.weight",
        ];
        assert!(!detect_moe(names).is_moe);
    }

    #[test]
    fn qwen_moe_style_triggers_detection() {
        let names = vec![
            "model.layers.0.mlp.gate.weight", // router
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
        ];
        let det = detect_moe(names);
        assert!(det.is_moe);
        assert_eq!(det.router_tensor_count, 1);
        assert_eq!(det.shared_expert_tensor_count, 1);
        assert!(det.expert_tensor_count >= 3);
    }

    #[test]
    fn unsupported_message_is_clear() {
        let det = detect_moe([
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.7.w1.weight",
        ]);
        let msg = unsupported_message(&det);
        assert!(msg.contains("MoE checkpoint detected"));
        assert!(msg.contains("not implemented yet"));
        assert!(msg.contains("experts=8"));
    }
}
