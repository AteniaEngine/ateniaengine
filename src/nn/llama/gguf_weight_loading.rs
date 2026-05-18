use crate::model_adapters::tensor_spec::{GEMMA2_SPEC, TransformParams, resolve_transforms};
use crate::nn::llama::config::LlamaConfig;
use crate::nn::llama::weight_loading::compute_transforms_for_name;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::{LoadTransform, WeightMapper};

pub fn llama_gguf_weight_mapper(
    config: &LlamaConfig,
    param_names: &[String],
    param_ids: &[usize],
) -> Result<WeightMapper, LoaderError> {
    let mut mapper = WeightMapper::from_param_names_and_ids(param_names, param_ids)?;
    let head_dim = config.head_dim();
    let kv_groups = config.kv_groups();
    let attention_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let hidden_size = config.hidden_size;

    for name in param_names {
        let transforms =
            llama_gguf_transforms_for_name(name, hidden_size, head_dim, kv_groups, attention_scale);
        if !transforms.is_empty() {
            mapper.set_transforms(name, transforms)?;
        }
    }
    Ok(mapper)
}

pub fn phi3_gguf_weight_mapper(
    config: &LlamaConfig,
    param_names: &[String],
    param_ids: &[usize],
) -> Result<WeightMapper, LoaderError> {
    let mut mapper = WeightMapper::from_param_names_and_ids(param_names, param_ids)?;
    let hidden_size = config.hidden_size;
    for name in param_names {
        let transforms = phi3_gguf_transforms_for_name(name, hidden_size);
        if !transforms.is_empty() {
            mapper.set_transforms(name, transforms)?;
        }
    }
    Ok(mapper)
}

pub fn gemma2_gguf_weight_mapper(
    config: &LlamaConfig,
    param_names: &[String],
    param_ids: &[usize],
) -> Result<WeightMapper, LoaderError> {
    let mut mapper = WeightMapper::from_param_names_and_ids(param_names, param_ids)?;
    let head_dim = config.effective_head_dim();
    let kv_groups = config.kv_groups();
    let qpas = config.query_pre_attn_scalar.unwrap_or(head_dim as f32);
    let attention_scale = 1.0_f32 / qpas.sqrt();
    let hidden_size = config.hidden_size;

    for name in param_names {
        let transforms = gemma2_gguf_transforms_for_name(
            name,
            hidden_size,
            head_dim,
            kv_groups,
            attention_scale,
        );
        if !transforms.is_empty() {
            mapper.set_transforms(name, transforms)?;
        }
    }
    Ok(mapper)
}

pub fn llama_gguf_transforms_for_name(
    name: &str,
    hidden_size: usize,
    head_dim: usize,
    kv_groups: usize,
    attention_scale: f32,
) -> Vec<LoadTransform> {
    let mut transforms = Vec::new();
    if name.contains(".self_attn.q_proj.weight") || name.contains(".self_attn.k_proj.weight") {
        transforms.push(LoadTransform::LlamaRopeUnpermuteRows { head_dim });
    }
    transforms.extend(compute_transforms_for_name(
        name,
        hidden_size,
        head_dim,
        kv_groups,
        attention_scale,
    ));
    transforms
}

/// Phi-3 GGUF per-name transforms.
///
/// The GGUF residency loader reverses the descriptor dimensions
/// (`weight_mapper::load_gguf_with_residency_plan`,
/// `descriptor.dimensions.iter().rev()`) before applying
/// transforms. After that reversal a Phi-3 GGUF tensor has the
/// **same logical orientation as its HF safetensors counterpart**
/// (Linear `[out, in]`, embed `[vocab, hidden]`, norms 1-D).
/// Therefore the correct GGUF transform set is identical to the
/// safetensors one, so this delegates to
/// [`crate::nn::llama::phi3::phi3_transforms_for_name`] — a single
/// source of truth. The previous bespoke table predated/ignored
/// the loader `.rev()` and was inverted (it transposed `embed`,
/// which must not be transposed, and skipped the Linear weights,
/// which must be — surfacing as the
/// `model.embed_tokens.weight: expected [32064,3072], got
/// [3072,32064]` shape mismatch).
pub fn phi3_gguf_transforms_for_name(name: &str, hidden_size: usize) -> Vec<LoadTransform> {
    crate::nn::llama::phi3::phi3_transforms_for_name(name, hidden_size)
}

/// **KNOWN BUG (GAP-1): pre-`.rev()` table — see ADR-006 / GAP-1.**
///
/// **AT-1c**: this bespoke Gemma 2 GGUF transform table is now
/// declarative data in `GEMMA2_SPEC.gguf_gap1_transforms`
/// ([`crate::model_adapters::tensor_spec::GEMMA2_SPEC`]),
/// preserved **verbatim and intentionally NOT corrected**. The
/// table is internally consistent with a world where the
/// residency loader does NOT reverse tensor dims (same defect
/// class as Phi-3 #4); the Gemma 2 GGUF path has never been
/// validated end-to-end. The correctness fix is the dedicated
/// post-AT-1 "Gemma 2 GGUF correctness" phase. Behaviour is
/// byte-identical to the previous hand-written ladder — pinned by
/// the AT-1a golden `golden_gemma2_gguf_gap1_matches_live_verbatim`
/// and the AT-2 snapshot
/// `gemma2_hf_and_gguf_tables_are_frozen_snapshots`.
pub fn gemma2_gguf_transforms_for_name(
    name: &str,
    hidden_size: usize,
    head_dim: usize,
    kv_groups: usize,
    attention_scale: f32,
) -> Vec<LoadTransform> {
    resolve_transforms(
        GEMMA2_SPEC
            .gguf_gap1_transforms
            .expect("GEMMA2_SPEC carries the GAP-1 verbatim GGUF table"),
        name,
        &TransformParams {
            hidden_size,
            head_dim,
            kv_groups,
            attention_scale,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_gguf_uses_hf_layout_transforms_after_dimension_reversal() {
        let t = llama_gguf_transforms_for_name("model.embed_tokens.weight", 2048, 64, 8, 0.125);
        assert!(t.is_empty());
        assert_eq!(
            llama_gguf_transforms_for_name(
                "model.layers.0.self_attn.q_proj.weight",
                2048,
                64,
                8,
                0.125,
            ),
            vec![
                LoadTransform::LlamaRopeUnpermuteRows { head_dim: 64 },
                LoadTransform::Transpose2D,
            ]
        );
    }

    #[test]
    fn llama_gguf_tiles_kv_on_output_dimension() {
        let k = llama_gguf_transforms_for_name(
            "model.layers.0.self_attn.k_proj.weight",
            2048,
            64,
            8,
            0.125,
        );
        assert_eq!(
            k,
            vec![
                LoadTransform::LlamaRopeUnpermuteRows { head_dim: 64 },
                LoadTransform::TileGroupedDim {
                    dim: 0,
                    group_size: 64,
                    repeats: 8,
                },
                LoadTransform::Transpose2D,
                LoadTransform::Scale { factor: 0.125 },
            ]
        );
        let v = llama_gguf_transforms_for_name(
            "model.layers.0.self_attn.v_proj.weight",
            2048,
            64,
            8,
            0.125,
        );
        assert_eq!(
            v,
            vec![
                LoadTransform::TileGroupedDim {
                    dim: 0,
                    group_size: 64,
                    repeats: 8,
                },
                LoadTransform::Transpose2D,
            ]
        );
    }

    #[test]
    fn gemma2_gguf_keeps_add_scalar_on_norms() {
        let t = gemma2_gguf_transforms_for_name(
            "model.layers.0.post_feedforward_layernorm.weight",
            2304,
            256,
            2,
            1.0 / 16.0,
        );
        assert_eq!(
            t,
            vec![
                LoadTransform::Reshape {
                    target: vec![1, 1, 2304],
                },
                LoadTransform::AddScalar { scalar: 1.0 },
            ]
        );
    }

    /// Phi-3 GGUF transforms must equal the safetensors table
    /// (single source of truth) and be correct per tensor class:
    /// embed → none; fused qkv / o / fused gate_up / down /
    /// lm_head → Transpose2D; norms → Reshape[1,1,hidden].
    #[test]
    fn phi3_gguf_transforms_match_safetensors_table() {
        use crate::nn::llama::phi3::phi3_transforms_for_name;
        let h = 3072;
        let names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
        ];
        for n in names {
            assert_eq!(
                phi3_gguf_transforms_for_name(n, h),
                phi3_transforms_for_name(n, h),
                "GGUF vs safetensors transform divergence for '{n}'"
            );
        }
        // Explicit per-class expectations (catches a regression in
        // the shared table too).
        assert_eq!(
            phi3_gguf_transforms_for_name("model.embed_tokens.weight", h),
            Vec::<LoadTransform>::new(),
            "embed must NOT be transposed (loader .rev already orients it)"
        );
        for n in [
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "lm_head.weight",
        ] {
            assert_eq!(
                phi3_gguf_transforms_for_name(n, h),
                vec![LoadTransform::Transpose2D],
                "Linear weight '{n}' must be Transpose2D"
            );
        }
        assert_eq!(
            phi3_gguf_transforms_for_name("model.layers.0.input_layernorm.weight", h),
            vec![LoadTransform::Reshape {
                target: vec![1, 1, h],
            }],
        );
    }
}
