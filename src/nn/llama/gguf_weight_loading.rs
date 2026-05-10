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

pub fn phi3_gguf_transforms_for_name(name: &str, hidden_size: usize) -> Vec<LoadTransform> {
    if name == "model.embed_tokens.weight" {
        return vec![LoadTransform::Transpose2D];
    }
    if name == "model.norm.weight" || name.ends_with("layernorm.weight") {
        return vec![LoadTransform::Reshape {
            target: vec![1, 1, hidden_size],
        }];
    }
    Vec::new()
}

pub fn gemma2_gguf_transforms_for_name(
    name: &str,
    hidden_size: usize,
    head_dim: usize,
    kv_groups: usize,
    attention_scale: f32,
) -> Vec<LoadTransform> {
    if name == "model.embed_tokens.weight" {
        return vec![LoadTransform::Transpose2D];
    }
    if name == "model.norm.weight" || name.ends_with("layernorm.weight") {
        return vec![
            LoadTransform::Reshape {
                target: vec![1, 1, hidden_size],
            },
            LoadTransform::AddScalar { scalar: 1.0 },
        ];
    }
    if name.contains(".self_attn.k_proj.weight") {
        return vec![
            LoadTransform::TileGroupedDim {
                dim: 1,
                group_size: head_dim,
                repeats: kv_groups,
            },
            LoadTransform::Scale {
                factor: attention_scale,
            },
        ];
    }
    if name.contains(".self_attn.v_proj.weight") {
        return vec![LoadTransform::TileGroupedDim {
            dim: 1,
            group_size: head_dim,
            repeats: kv_groups,
        }];
    }
    Vec::new()
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
}
