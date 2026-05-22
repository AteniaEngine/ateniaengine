use crate::nn::llama::config::{LlamaConfig, ROPE_THETA_LEGACY_DEFAULT};
use crate::v17::loader::gguf_reader::{GgufError, GgufReader, MetadataValue};

pub fn architecture_from_gguf(reader: &GgufReader) -> Result<String, GgufError> {
    let arch = required_string(reader, "general.architecture")?;
    match arch {
        "llama" => Ok("LlamaForCausalLM".to_string()),
        "qwen2" => Ok("Qwen2ForCausalLM".to_string()),
        "qwen3" => Ok("Qwen3ForCausalLM".to_string()),
        "phi3" => Ok("Phi3ForCausalLM".to_string()),
        "gemma2" => Ok("Gemma2ForCausalLM".to_string()),
        "gemma3" => Ok("Gemma3ForCausalLM".to_string()),
        other => Err(GgufError::InvalidFormat(format!(
            "GGUF config: unsupported general.architecture = \"{other}\"; \
             Atenia supports llama, qwen2, qwen3, phi3, gemma2, and gemma3"
        ))),
    }
}

pub fn llama_config_from_gguf(reader: &GgufReader) -> Result<LlamaConfig, GgufError> {
    let arch = required_string(reader, "general.architecture")?;
    let prefix = arch_prefix(arch)?;
    // NOTE: this holds num_key_value_heads (the GGUF
    // `attention.head_count_kv`), NOT a head *dimension*. Renamed
    // in G-1b so it is not confused with the real head_dim fix
    // immediately below (it was previously named `head_dim`).
    let num_kv_heads = optional_usize(reader, &format!("{prefix}.attention.head_count_kv"));
    let hidden_size = required_usize(reader, &format!("{prefix}.embedding_length"))?;
    let num_attention_heads = required_usize(reader, &format!("{prefix}.attention.head_count"))?;
    // **G-1b / GAP-C1** — the explicit head dimension is the
    // llama.cpp-standard `attention.key_length` (Gemma 2 sets it
    // to 256 while hidden/heads = 288). The old `attention.head_size`
    // key is not a llama.cpp convention and was never present, so
    // head_dim silently fell back to the (wrong) inferred value for
    // any model where head_dim != hidden/heads. Behaviour-preserving
    // for every previously-validated GGUF (TinyLlama / SmolLM2 /
    // Llama-3.2 / Phi-3.5 all have key_length == hidden/heads, so
    // the `!= inferred` filter below still yields None there).
    let explicit_head_dim = optional_usize(reader, &format!("{prefix}.attention.key_length"));
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
        num_key_value_heads: num_kv_heads.unwrap_or(num_attention_heads),
        intermediate_size: required_usize(reader, &format!("{prefix}.feed_forward_length"))?,
        max_position_embeddings: required_usize(reader, &format!("{prefix}.context_length"))?,
        rope_theta,
        rms_norm_eps: required_f32(
            reader,
            &format!("{prefix}.attention.layer_norm_rms_epsilon"),
        )?,
        tie_word_embeddings,
        // **Qwen GGUF support** — `None` (not `Some(false)`) so the
        // resolved adapter owns the family default via
        // `effective_attention_bias()`: Qwen2 GGUF carries QKV
        // biases (`Qwen2Adapter::default_attention_bias = Some(true)`),
        // while Llama / Qwen3 / Phi-3 / Gemma 2 have none (their
        // adapter default `None` resolves to `false`). Behaviour is
        // byte-identical to the old hard-coded `Some(false)` for
        // every previously-supported family; only Qwen2 changes,
        // and there the old value was wrong.
        attention_bias: None,
        model_type: Some(model_type_for_arch(arch).to_string()),
        bos_token_id: optional_u32(reader, "tokenizer.ggml.bos_token_id").unwrap_or(1),
        eos_token_id: optional_u32(reader, "tokenizer.ggml.eos_token_id").unwrap_or(2),
        // GGUF metadata carries a single `eos_token_id` scalar
        // (llama.cpp picks the family-correct one — e.g.
        // `<end_of_turn>` for Gemma instruct), so the multi-EOS
        // set is just that scalar.
        eos_token_ids: vec![optional_u32(reader, "tokenizer.ggml.eos_token_id").unwrap_or(2)],
        pad_token_id: optional_u32(reader, "tokenizer.ggml.padding_token_id"),
        head_dim: explicit_head_dim.filter(|d| *d != inferred_head_dim),
        rope_scaling: None,
        // **G-1b / GAP-C2** — read the real llama.cpp softcap keys
        // (`{arch}.attn_logit_softcapping` / `final_logit_softcapping`
        // = 50 / 30 for Gemma 2). The old `attention.logit_softcap`
        // / `final_logit_softcap` keys never matched a real
        // checkpoint, so the values only ever came from the
        // Gemma2Adapter 50/30 fallback. With the correct keys the
        // adapter default applies only when the GGUF omits them.
        attn_logit_softcapping: optional_f32(
            reader,
            &format!("{prefix}.attn_logit_softcapping"),
        ),
        final_logit_softcapping: optional_f32(
            reader,
            &format!("{prefix}.final_logit_softcapping"),
        ),
        sliding_window: optional_u32(reader, &format!("{prefix}.attention.sliding_window")),
        query_pre_attn_scalar: optional_f32(
            reader,
            &format!("{prefix}.attention.query_pre_attn_scalar"),
        ),
        // **Gemma 3** — dual-RoPE / sliding-window-pattern GGUF
        // metadata. llama.cpp's `LLM_ARCH_GEMMA3` stores the local
        // (sliding) RoPE base under `gemma3.rope.local.freq_base`
        // and the global one under the standard
        // `gemma3.rope.freq_base` (mapped to `rope_theta` above);
        // the local:global period is `gemma3.attention
        // .sliding_window_pattern`. `None` for every non-Gemma-3
        // GGUF, so `cfg` stays byte-equivalent for those.
        rope_local_base_freq: optional_f32(reader, &format!("{prefix}.rope.local.freq_base"))
            .map(|v| v as u32),
        sliding_window_pattern: optional_u32(
            reader,
            &format!("{prefix}.attention.sliding_window_pattern"),
        ),
        // **Phi-4 (partial rotary).** A Phi-4 GGUF stores the
        // rotated dimension count as the absolute
        // `{arch}.rope.dimension_count`; convert it back to the
        // HF-style fraction `dimension_count / head_dim`. `None`
        // when the key is absent (every full-rotary GGUF, where
        // dimension_count == head_dim anyway) — kept `None` then so
        // `rotary_dim()` returns the full head_dim unchanged.
        partial_rotary_factor: optional_usize(reader, &format!("{prefix}.rope.dimension_count"))
            .and_then(|dim| {
                let head_dim = explicit_head_dim.unwrap_or(inferred_head_dim);
                if head_dim > 0 && dim != head_dim {
                    Some(dim as f32 / head_dim as f32)
                } else {
                    None
                }
            }),
    };

    // **Phase 15** — the resolved adapter owns family config
    // policy for both formats. The GGUF parser only translates
    // its rope-scaling metadata into the same HF-shaped JSON the
    // adapter already consumes (Phi-3 LongRope via
    // `Phi3Adapter::parse_rope_scaling`, Phase 12.4); the
    // Gemma 2 softcaps and the Phi-3/Gemma-2 explicit-head_dim
    // policy move to `apply_config_defaults`. `GgufReader` never
    // crosses into the adapter layer. `resolve_adapter_for_config`
    // returns a `'static` reference, so `cfg` stays free to be
    // mutated below.
    let adapter = crate::model_adapters::resolve_adapter_for_config(&cfg);
    if let Some(rope_json) = gguf_rope_scaling_json(reader, prefix, cfg.max_position_embeddings)? {
        cfg.rope_scaling = adapter
            .parse_rope_scaling(&rope_json)
            .map_err(|e| GgufError::InvalidFormat(format!("GGUF config rope_scaling: {e}")))?;
    }
    adapter.apply_config_defaults(&mut cfg);

    cfg.validate()
        .map_err(|e| GgufError::InvalidFormat(format!("GGUF config validation failed: {e}")))?;
    // **Phase 13** — family-specific config validation, parity
    // with the safetensors `LlamaConfig::from_json_str` path.
    adapter
        .validate_config(&cfg)
        .map_err(|e| GgufError::InvalidFormat(format!("GGUF config validation failed: {e}")))?;
    Ok(cfg)
}

fn arch_prefix(arch: &str) -> Result<&str, GgufError> {
    match arch {
        "llama" | "qwen2" | "qwen3" | "phi3" | "gemma2" | "gemma3" => Ok(arch),
        other => Err(GgufError::InvalidFormat(format!(
            "GGUF config: unsupported general.architecture = \"{other}\""
        ))),
    }
}

fn model_type_for_arch(arch: &str) -> &'static str {
    match arch {
        "qwen2" => "qwen2",
        "qwen3" => "qwen3",
        "phi3" => "phi3",
        "gemma2" => "gemma2",
        "gemma3" => "gemma3_text",
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

/// Resolve one LongRope factor vector. Phi-3 GGUFs use one of
/// two interchangeable encodings:
///
/// 1. **metadata-array** — `{prefix}.rope.scaling.{short,long}_factor`
///    as f32 metadata arrays (the form Phase 15 originally
///    handled);
/// 2. **tensor** — top-level `rope_factors_{short,long}.weight`
///    F32 tensors (the canonical modern llama.cpp Phi-3 layout,
///    e.g. `phi-3.5-mini-instruct-q4_k_m.gguf`).
///
/// The metadata form is tried first (cheap, no file I/O); on
/// absence it falls back to decoding the tensor form via the
/// existing [`crate::v17::loader::gguf_decode::decode_tensor`].
/// `None` means *neither* form is present — a non-LongRope GGUF
/// (Llama / Gemma 2), preserved exactly as before.
fn longrope_factor(
    reader: &GgufReader,
    meta_key: &str,
    tensor_name: &str,
) -> Result<Option<Vec<f32>>, GgufError> {
    if let Some(v) = f32_array(reader, meta_key)? {
        return Ok(Some(v));
    }
    let Some(descriptor) = reader.tensor_by_name(tensor_name) else {
        return Ok(None);
    };
    let v = crate::v17::loader::gguf_decode::decode_tensor(reader, descriptor)?;
    Ok(Some(v))
}

/// **Phase 15** — translate GGUF rope-scaling into the HF-shaped
/// `serde_json::Value` that
/// [`crate::model_adapters::ConfigPolicy::parse_rope_scaling`]
/// consumes. Pure format translation: GGUF encodes LongRope by
/// the presence of short/long factors (metadata-array *or* the
/// `rope_factors_{short,long}.weight` tensor form — see
/// [`longrope_factor`]); HF encodes it via an explicit
/// `rope_scaling.type = "longrope"` discriminator plus top-level
/// position-embedding fields. The *semantics* (which family owns
/// longrope, building `RopeScaling::LongRope`, factor-length
/// validation) stay in `Phi3Adapter::parse_rope_scaling`. Returns
/// `None` when the GGUF carries no rope-scaling factors in either
/// form — the common case for Llama / Gemma 2 GGUF, exactly as
/// the pre-Phase-15 `if arch == "phi3"` gate behaved by never
/// invoking this for other families.
fn gguf_rope_scaling_json(
    reader: &GgufReader,
    prefix: &str,
    max_position_embeddings: usize,
) -> Result<Option<serde_json::Value>, GgufError> {
    let Some(short_factor) = longrope_factor(
        reader,
        &format!("{prefix}.rope.scaling.short_factor"),
        "rope_factors_short.weight",
    )?
    else {
        return Ok(None);
    };
    let long_factor = longrope_factor(
        reader,
        &format!("{prefix}.rope.scaling.long_factor"),
        "rope_factors_long.weight",
    )?
    .ok_or_else(|| {
        GgufError::InvalidFormat(
            "GGUF config: LongRope short_factor present but long_factor missing".to_string(),
        )
    })?;
    let original_max_position_embeddings = optional_u32(
        reader,
        &format!("{prefix}.rope.scaling.original_context_length"),
    )
    .unwrap_or(max_position_embeddings as u32);
    Ok(Some(serde_json::json!({
        "rope_scaling": {
            "type": "longrope",
            "short_factor": short_factor,
            "long_factor": long_factor,
        },
        "original_max_position_embeddings": original_max_position_embeddings,
        "max_position_embeddings": max_position_embeddings as u32,
    })))
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

    /// Small synthetic reader for non-Llama families. `prefix`
    /// is the GGUF arch (`"phi3"` / `"gemma2"`); dims kept tiny
    /// (hidden 64, 4 heads → head_dim 16) so the structural
    /// validators pass without large fixtures. Family-specific
    /// metadata (rope scaling, softcaps) is inserted by the
    /// caller via `reader.metadata`.
    fn small_family_reader(arch: &str) -> GgufReader {
        let mut metadata = HashMap::new();
        let p = |k: &str| format!("{arch}.{k}");
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String(arch.to_string()),
        );
        metadata.insert(p("context_length"), MetadataValue::UInt32(4096));
        metadata.insert(p("embedding_length"), MetadataValue::UInt32(64));
        metadata.insert(p("block_count"), MetadataValue::UInt32(2));
        metadata.insert(p("attention.head_count"), MetadataValue::UInt32(4));
        metadata.insert(p("attention.head_count_kv"), MetadataValue::UInt32(4));
        metadata.insert(p("feed_forward_length"), MetadataValue::UInt32(128));
        metadata.insert(
            p("attention.layer_norm_rms_epsilon"),
            MetadataValue::Float32(1e-6),
        );
        metadata.insert(p("rope.freq_base"), MetadataValue::Float32(10000.0));
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
                values: (0..256)
                    .map(|i| MetadataValue::String(format!("tok{i}")))
                    .collect(),
            }),
        );
        let tensors = vec![TensorDescriptor {
            name: "token_embd.weight".to_string(),
            dimensions: vec![64, 256],
            tensor_type: GgufTensorType::Q8_0,
            offset: 0,
        }];
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

    fn f32_meta_array(values: &[f32]) -> MetadataValue {
        MetadataValue::Array(MetadataArray {
            element_type: MetadataType::Float32,
            values: values.iter().map(|v| MetadataValue::Float32(*v)).collect(),
        })
    }

    /// **Phase 15** — Phi-3 GGUF LongRope: the GGUF parser
    /// translates `phi3.rope.scaling.*` metadata into HF-shaped
    /// JSON, and `Phi3Adapter::parse_rope_scaling` (Phase 12.4)
    /// builds `RopeScaling::LongRope`. `GgufReader` never enters
    /// the adapter; the family semantics are adapter-owned.
    #[test]
    fn phase15_phi3_gguf_longrope_via_adapter() {
        use crate::nn::llama::config::RopeScaling;
        let mut reader = small_family_reader("phi3");
        let short: Vec<f32> = (0..8).map(|i| 1.0 + i as f32 * 0.01).collect();
        let long: Vec<f32> = (0..8).map(|i| 2.0 + i as f32 * 0.05).collect();
        reader
            .metadata
            .insert("phi3.rope.scaling.short_factor".to_string(), f32_meta_array(&short));
        reader
            .metadata
            .insert("phi3.rope.scaling.long_factor".to_string(), f32_meta_array(&long));
        reader.metadata.insert(
            "phi3.rope.scaling.original_context_length".to_string(),
            MetadataValue::UInt32(4096),
        );

        let cfg = llama_config_from_gguf(&reader).expect("phi3 GGUF LongRope config must parse");
        assert_eq!(cfg.model_type.as_deref(), Some("phi3"));
        match cfg.rope_scaling {
            Some(RopeScaling::LongRope {
                ref short_factor,
                ref long_factor,
                original_max_position_embeddings,
                max_position_embeddings,
            }) => {
                assert_eq!(short_factor.len(), 8);
                assert_eq!(long_factor.len(), 8);
                assert!((short_factor[0] - 1.0).abs() < 1e-6);
                assert!((long_factor[0] - 2.0).abs() < 1e-6);
                assert_eq!(original_max_position_embeddings, 4096);
                assert_eq!(max_position_embeddings, 4096);
            }
            other => panic!("expected RopeScaling::LongRope, got {other:?}"),
        }
        // Phi3Adapter::apply_config_defaults makes head_dim
        // explicit (= effective_head_dim = 64 / 4).
        assert_eq!(cfg.head_dim, Some(16));
    }

    /// **Phi-3 GGUF LongRope, tensor encoding.** The canonical
    /// modern llama.cpp Phi-3 GGUF (e.g.
    /// `phi-3.5-mini-instruct-q4_k_m.gguf`) stores the LongRope
    /// factors as top-level `rope_factors_{short,long}.weight`
    /// F32 tensors, *not* as `phi3.rope.scaling.{short,long}_factor`
    /// metadata arrays. `gguf_rope_scaling_json` must fall back to
    /// decoding those tensors. Uses a minimal on-disk fixture
    /// because `decode_tensor` reopens `reader.file_path`.
    #[test]
    fn phi3_gguf_longrope_via_tensor_encoding() {
        use crate::nn::llama::config::RopeScaling;

        let short: Vec<f32> = (0..8).map(|i| 1.0 + i as f32 * 0.01).collect();
        let long: Vec<f32> = (0..8).map(|i| 2.0 + i as f32 * 0.05).collect();

        // Fixture file = [short f32 LE][long f32 LE], contiguous.
        let mut bytes = Vec::with_capacity(64);
        for v in &short {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in &long {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let path = std::env::temp_dir().join(format!(
            "atenia_phi3_longrope_tensor_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write longrope tensor fixture");

        let mut reader = small_family_reader("phi3");
        // Tensor form only — NO metadata short/long_factor arrays.
        reader.metadata.insert(
            "phi3.rope.scaling.original_context_length".to_string(),
            MetadataValue::UInt32(4096),
        );
        reader.data_section_offset = 0;
        reader.file_path = path.clone();
        reader.tensors.push(TensorDescriptor {
            name: "rope_factors_short.weight".to_string(),
            dimensions: vec![8],
            tensor_type: GgufTensorType::F32,
            offset: 0,
        });
        reader.tensors.push(TensorDescriptor {
            name: "rope_factors_long.weight".to_string(),
            dimensions: vec![8],
            tensor_type: GgufTensorType::F32,
            offset: (short.len() * 4) as u64,
        });

        let parsed = llama_config_from_gguf(&reader);
        // decode_tensor already read the bytes during parse;
        // remove the fixture before asserting so a failed assert
        // does not leak the temp file.
        let _ = std::fs::remove_file(&path);
        let cfg = parsed.expect("phi3 GGUF tensor-encoded LongRope must parse");

        assert_eq!(cfg.model_type.as_deref(), Some("phi3"));
        match cfg.rope_scaling {
            Some(RopeScaling::LongRope {
                ref short_factor,
                ref long_factor,
                original_max_position_embeddings,
                max_position_embeddings,
            }) => {
                assert_eq!(short_factor.len(), 8);
                assert_eq!(long_factor.len(), 8);
                assert!((short_factor[0] - 1.0).abs() < 1e-6);
                assert!((short_factor[7] - 1.07).abs() < 1e-5);
                assert!((long_factor[0] - 2.0).abs() < 1e-6);
                assert!((long_factor[7] - 2.35).abs() < 1e-5);
                assert_eq!(original_max_position_embeddings, 4096);
                assert_eq!(max_position_embeddings, 4096);
            }
            other => panic!("expected RopeScaling::LongRope, got {other:?}"),
        }
    }

    /// **Phase 15** — Gemma 2 GGUF defaults: with no softcap /
    /// head_dim metadata, `Gemma2Adapter::apply_config_defaults`
    /// injects the family constants (50/30, head_dim,
    /// query_pre_attn_scalar). Behaviour-equivalent to the
    /// removed `if arch == "gemma2"` block.
    #[test]
    fn phase15_gemma2_gguf_defaults_via_adapter() {
        let reader = small_family_reader("gemma2");
        let cfg = llama_config_from_gguf(&reader).expect("gemma2 GGUF config must parse");
        assert_eq!(cfg.model_type.as_deref(), Some("gemma2"));
        assert_eq!(cfg.head_dim, Some(16)); // effective = 64 / 4
        assert_eq!(cfg.attn_logit_softcapping, Some(50.0));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert_eq!(cfg.query_pre_attn_scalar, Some(16.0));
        assert!(cfg.rope_scaling.is_none());
    }

    /// **G-1b / GAP-C1 + GAP-C2** — with the *real* llama.cpp
    /// Gemma 2 metadata keys present (bartowski/gemma-2-2b-it
    /// Q4_K_M shape), head_dim / query_pre_attn_scalar / softcaps
    /// must be sourced from the GGUF, not from the adapter's
    /// 50/30/effective-head_dim fallback. head_dim = 256 is the
    /// proof for GAP-C1: it differs from hidden/heads = 2304/8 =
    /// 288, so it can only have come from `attention.key_length`.
    #[test]
    fn gemma2_gguf_config_reads_real_metadata_keys() {
        let mut reader = small_family_reader("gemma2");
        reader.metadata.insert(
            "gemma2.embedding_length".to_string(),
            MetadataValue::UInt32(2304),
        );
        reader.metadata.insert(
            "gemma2.attention.head_count".to_string(),
            MetadataValue::UInt32(8),
        );
        reader.metadata.insert(
            "gemma2.attention.head_count_kv".to_string(),
            MetadataValue::UInt32(4),
        );
        reader.metadata.insert(
            "gemma2.attention.key_length".to_string(),
            MetadataValue::UInt32(256),
        );
        reader.metadata.insert(
            "gemma2.attention.value_length".to_string(),
            MetadataValue::UInt32(256),
        );
        reader.metadata.insert(
            "gemma2.attn_logit_softcapping".to_string(),
            MetadataValue::Float32(50.0),
        );
        reader.metadata.insert(
            "gemma2.final_logit_softcapping".to_string(),
            MetadataValue::Float32(30.0),
        );

        let cfg =
            llama_config_from_gguf(&reader).expect("gemma2 real-keys GGUF config must parse");
        assert_eq!(cfg.hidden_size, 2304);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 4);
        // GAP-C1: from attention.key_length (256), NOT hidden/heads
        // (288) and NOT the adapter default.
        assert_eq!(cfg.head_dim, Some(256));
        // qpas chains off the (now correct) head_dim.
        assert_eq!(cfg.query_pre_attn_scalar, Some(256.0));
        // GAP-C2: from the real GGUF softcap keys.
        assert_eq!(cfg.attn_logit_softcapping, Some(50.0));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));

        // Prove the softcaps are GGUF-sourced, not the adapter's
        // hard 50/30: non-default values must pass through (the
        // adapter fallback could only ever produce 50/30).
        let mut r2 = small_family_reader("gemma2");
        r2.metadata.insert(
            "gemma2.attention.key_length".to_string(),
            MetadataValue::UInt32(256),
        );
        r2.metadata.insert(
            "gemma2.attn_logit_softcapping".to_string(),
            MetadataValue::Float32(99.0),
        );
        r2.metadata.insert(
            "gemma2.final_logit_softcapping".to_string(),
            MetadataValue::Float32(88.0),
        );
        let cfg2 = llama_config_from_gguf(&r2).expect("config");
        assert_eq!(
            cfg2.attn_logit_softcapping,
            Some(99.0),
            "softcap must come from the GGUF key, not the adapter 50/30 default"
        );
        assert_eq!(cfg2.final_logit_softcapping, Some(88.0));
    }
}
