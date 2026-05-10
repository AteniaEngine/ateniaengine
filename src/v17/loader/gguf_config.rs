use crate::nn::llama::config::{LlamaConfig, ROPE_THETA_LEGACY_DEFAULT, RopeScaling};
use crate::v17::loader::gguf_reader::{GgufError, GgufReader, MetadataValue};

pub fn architecture_from_gguf(reader: &GgufReader) -> Result<String, GgufError> {
    let arch = required_string(reader, "general.architecture")?;
    match arch {
        "llama" => Ok("LlamaForCausalLM".to_string()),
        "phi3" => Ok("Phi3ForCausalLM".to_string()),
        "gemma2" => Ok("Gemma2ForCausalLM".to_string()),
        other => Err(GgufError::InvalidFormat(format!(
            "GGUF config: unsupported general.architecture = \"{other}\"; \
             Atenia supports llama, phi3, and gemma2 in M11.D.3"
        ))),
    }
}

pub fn llama_config_from_gguf(reader: &GgufReader) -> Result<LlamaConfig, GgufError> {
    let arch = required_string(reader, "general.architecture")?;
    let prefix = arch_prefix(arch)?;
    let head_dim = optional_usize(reader, &format!("{prefix}.attention.head_count_kv"));
    let hidden_size = required_usize(reader, &format!("{prefix}.embedding_length"))?;
    let num_attention_heads = required_usize(reader, &format!("{prefix}.attention.head_count"))?;
    let explicit_head_dim = optional_usize(reader, &format!("{prefix}.attention.head_size"));
    let inferred_head_dim = hidden_size / num_attention_heads;
    let rope_theta = optional_f32(reader, &format!("{prefix}.rope.freq_base"))
        .map(|v| v as u32)
        .unwrap_or(ROPE_THETA_LEGACY_DEFAULT);
    let tie_word_embeddings = reader.tensor_by_name("output.weight").is_none();

    let mut cfg = LlamaConfig {
        vocab_size: vocab_size_from_metadata(reader)?,
        hidden_size,
        num_hidden_layers: required_usize(reader, &format!("{prefix}.block_count"))?,
        num_attention_heads,
        num_key_value_heads: head_dim.unwrap_or(num_attention_heads),
        intermediate_size: required_usize(reader, &format!("{prefix}.feed_forward_length"))?,
        max_position_embeddings: required_usize(reader, &format!("{prefix}.context_length"))?,
        rope_theta,
        rms_norm_eps: required_f32(
            reader,
            &format!("{prefix}.attention.layer_norm_rms_epsilon"),
        )?,
        tie_word_embeddings,
        attention_bias: Some(false),
        model_type: Some(model_type_for_arch(arch).to_string()),
        bos_token_id: optional_u32(reader, "tokenizer.ggml.bos_token_id").unwrap_or(1),
        eos_token_id: optional_u32(reader, "tokenizer.ggml.eos_token_id").unwrap_or(2),
        pad_token_id: optional_u32(reader, "tokenizer.ggml.padding_token_id"),
        head_dim: explicit_head_dim.filter(|d| *d != inferred_head_dim),
        rope_scaling: None,
        attn_logit_softcapping: optional_f32(reader, &format!("{prefix}.attention.logit_softcap")),
        final_logit_softcapping: optional_f32(reader, &format!("{prefix}.final_logit_softcap")),
        sliding_window: optional_u32(reader, &format!("{prefix}.attention.sliding_window")),
        query_pre_attn_scalar: optional_f32(
            reader,
            &format!("{prefix}.attention.query_pre_attn_scalar"),
        ),
    };

    if arch == "phi3" {
        cfg.rope_scaling = longrope_from_gguf(reader, prefix, cfg.max_position_embeddings)?;
        cfg.head_dim = Some(explicit_head_dim.unwrap_or(inferred_head_dim));
    }
    if arch == "gemma2" {
        cfg.head_dim = Some(explicit_head_dim.unwrap_or(inferred_head_dim));
        cfg.attn_logit_softcapping = cfg.attn_logit_softcapping.or(Some(50.0));
        cfg.final_logit_softcapping = cfg.final_logit_softcapping.or(Some(30.0));
        cfg.query_pre_attn_scalar = cfg.query_pre_attn_scalar.or(cfg.head_dim.map(|d| d as f32));
    }

    cfg.validate()
        .map_err(|e| GgufError::InvalidFormat(format!("GGUF config validation failed: {e}")))?;
    Ok(cfg)
}

fn arch_prefix(arch: &str) -> Result<&str, GgufError> {
    match arch {
        "llama" | "phi3" | "gemma2" => Ok(arch),
        other => Err(GgufError::InvalidFormat(format!(
            "GGUF config: unsupported general.architecture = \"{other}\""
        ))),
    }
}

fn model_type_for_arch(arch: &str) -> &'static str {
    match arch {
        "phi3" => "phi3",
        "gemma2" => "gemma2",
        _ => "llama",
    }
}

fn metadata<'a>(reader: &'a GgufReader, key: &str) -> Option<&'a MetadataValue> {
    reader.metadata.get(key)
}

fn required_string<'a>(reader: &'a GgufReader, key: &str) -> Result<&'a str, GgufError> {
    metadata(reader, key)
        .and_then(|v| v.as_string())
        .ok_or_else(|| {
            GgufError::InvalidFormat(format!(
                "GGUF config: missing or non-string metadata key '{key}'"
            ))
        })
}

fn required_usize(reader: &GgufReader, key: &str) -> Result<usize, GgufError> {
    metadata(reader, key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| {
            GgufError::InvalidFormat(format!(
                "GGUF config: missing or non-integer metadata key '{key}'"
            ))
        })
}

fn required_f32(reader: &GgufReader, key: &str) -> Result<f32, GgufError> {
    metadata(reader, key)
        .and_then(|v| v.as_f32())
        .ok_or_else(|| {
            GgufError::InvalidFormat(format!(
                "GGUF config: missing or non-numeric metadata key '{key}'"
            ))
        })
}

fn optional_usize(reader: &GgufReader, key: &str) -> Option<usize> {
    metadata(reader, key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
}

fn optional_u32(reader: &GgufReader, key: &str) -> Option<u32> {
    metadata(reader, key)
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
}

fn optional_f32(reader: &GgufReader, key: &str) -> Option<f32> {
    metadata(reader, key).and_then(|v| v.as_f32())
}

fn vocab_size_from_metadata(reader: &GgufReader) -> Result<usize, GgufError> {
    match metadata(reader, "tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(arr)) => Ok(arr.values.len()),
        _ => Err(GgufError::InvalidFormat(
            "GGUF config: missing tokenizer.ggml.tokens array for vocab_size".to_string(),
        )),
    }
}

fn f32_array(reader: &GgufReader, key: &str) -> Result<Option<Vec<f32>>, GgufError> {
    let Some(MetadataValue::Array(arr)) = metadata(reader, key) else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(arr.values.len());
    for (idx, value) in arr.values.iter().enumerate() {
        let Some(v) = value.as_f32() else {
            return Err(GgufError::InvalidFormat(format!(
                "GGUF config: metadata key '{key}' element {idx} is not numeric"
            )));
        };
        out.push(v);
    }
    Ok(Some(out))
}

fn longrope_from_gguf(
    reader: &GgufReader,
    prefix: &str,
    max_position_embeddings: usize,
) -> Result<Option<RopeScaling>, GgufError> {
    let Some(short_factor) = f32_array(reader, &format!("{prefix}.rope.scaling.short_factor"))?
    else {
        return Ok(None);
    };
    let long_factor = f32_array(reader, &format!("{prefix}.rope.scaling.long_factor"))?
        .ok_or_else(|| {
            GgufError::InvalidFormat(
                "GGUF config: phi3 LongRope short_factor present but long_factor missing"
                    .to_string(),
            )
        })?;
    let original_max_position_embeddings = optional_u32(
        reader,
        &format!("{prefix}.rope.scaling.original_context_length"),
    )
    .unwrap_or(max_position_embeddings as u32);
    Ok(Some(RopeScaling::LongRope {
        short_factor,
        long_factor,
        original_max_position_embeddings,
        max_position_embeddings: max_position_embeddings as u32,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v17::loader::gguf_reader::{
        GgufReader, GgufTensorType, MetadataArray, MetadataType, TensorDescriptor,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn tiny_reader(has_output: bool) -> GgufReader {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String("llama".to_string()),
        );
        metadata.insert(
            "llama.context_length".to_string(),
            MetadataValue::UInt32(2048),
        );
        metadata.insert(
            "llama.embedding_length".to_string(),
            MetadataValue::UInt32(2048),
        );
        metadata.insert("llama.block_count".to_string(), MetadataValue::UInt32(22));
        metadata.insert(
            "llama.attention.head_count".to_string(),
            MetadataValue::UInt32(32),
        );
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            MetadataValue::UInt32(4),
        );
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            MetadataValue::UInt32(5632),
        );
        metadata.insert(
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            MetadataValue::Float32(1e-5),
        );
        metadata.insert(
            "llama.rope.freq_base".to_string(),
            MetadataValue::Float32(10000.0),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            MetadataValue::UInt32(1),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            MetadataValue::UInt32(2),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            MetadataValue::Array(MetadataArray {
                element_type: MetadataType::String,
                values: (0..32_000)
                    .map(|i| MetadataValue::String(format!("tok{i}")))
                    .collect(),
            }),
        );
        let mut tensors = vec![TensorDescriptor {
            name: "token_embd.weight".to_string(),
            dimensions: vec![2048, 32_000],
            tensor_type: GgufTensorType::Q8_0,
            offset: 0,
        }];
        if has_output {
            tensors.push(TensorDescriptor {
                name: "output.weight".to_string(),
                dimensions: vec![2048, 32_000],
                tensor_type: GgufTensorType::Q8_0,
                offset: 0,
            });
        }
        GgufReader {
            version: 3,
            tensor_count: tensors.len() as u64,
            metadata,
            tensors,
            data_section_offset: 32,
            alignment: 32,
            file_path: PathBuf::from("synthetic.gguf"),
        }
    }

    #[test]
    fn detects_architecture() {
        let reader = tiny_reader(false);
        assert_eq!(architecture_from_gguf(&reader).unwrap(), "LlamaForCausalLM");
    }

    #[test]
    fn builds_tinyllama_config_and_tied_detection() {
        let reader = tiny_reader(false);
        let cfg = llama_config_from_gguf(&reader).expect("config");
        assert_eq!(cfg.vocab_size, 32_000);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 22);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.model_type.as_deref(), Some("llama"));
    }

    #[test]
    fn detects_untied_when_output_weight_exists() {
        let reader = tiny_reader(true);
        let cfg = llama_config_from_gguf(&reader).expect("config");
        assert!(!cfg.tie_word_embeddings);
    }
}
