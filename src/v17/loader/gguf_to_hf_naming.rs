/// Split a `blk.<layer>.<suffix>` GGUF tensor name into the HF
/// layer prefix (`model.layers.<layer>`) and the raw suffix.
/// Returns `None` for non-block names or a non-numeric layer.
fn split_blk(gguf_name: &str) -> Option<(String, &str)> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    if layer.is_empty() || !layer.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((format!("model.layers.{layer}"), suffix))
}

/// **Phase 16.1** — architecture-agnostic GGUF → HF name mapping:
/// the top-level tensors plus the common Llama-layout block
/// suffixes shared by every supported family. Returns `None` for
/// family-specific suffixes (handled by the `*_gguf_extra`
/// functions) and for unknown / malformed names.
pub fn gguf_to_hf_name_common(gguf_name: &str) -> Option<String> {
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        "rope_freqs.weight" => return Some("rope_freqs".to_string()),
        _ => {}
    }

    let (prefix, suffix) = split_blk(gguf_name)?;
    match suffix {
        "attn_norm.weight" => Some(format!("{prefix}.input_layernorm.weight")),
        "attn_q.weight" => Some(format!("{prefix}.self_attn.q_proj.weight")),
        "attn_k.weight" => Some(format!("{prefix}.self_attn.k_proj.weight")),
        "attn_v.weight" => Some(format!("{prefix}.self_attn.v_proj.weight")),
        "attn_output.weight" => Some(format!("{prefix}.self_attn.o_proj.weight")),
        "ffn_norm.weight" => Some(format!("{prefix}.post_attention_layernorm.weight")),
        "ffn_gate.weight" => Some(format!("{prefix}.mlp.gate_proj.weight")),
        "ffn_up.weight" => Some(format!("{prefix}.mlp.up_proj.weight")),
        "ffn_down.weight" => Some(format!("{prefix}.mlp.down_proj.weight")),
        _ => None,
    }
}

/// **Phase 16.1** — Phi-3-specific GGUF suffix(es) not covered by
/// [`gguf_to_hf_name_common`]: the fused QKV projection.
pub fn phi3_gguf_extra(gguf_name: &str) -> Option<String> {
    let (prefix, suffix) = split_blk(gguf_name)?;
    match suffix {
        "attn_qkv.weight" => Some(format!("{prefix}.self_attn.qkv_proj.weight")),
        _ => None,
    }
}

/// **Phase 16.1** — Gemma 2-specific GGUF suffix(es) not covered by
/// [`gguf_to_hf_name_common`]: the post-attention / post-FFN norms.
pub fn gemma2_gguf_extra(gguf_name: &str) -> Option<String> {
    let (prefix, suffix) = split_blk(gguf_name)?;
    match suffix {
        "attn_post_norm.weight" => Some(format!("{prefix}.post_attention_layernorm.weight")),
        "post_attention_norm.weight" => {
            Some(format!("{prefix}.post_attention_layernorm.weight"))
        }
        "ffn_post_norm.weight" => Some(format!("{prefix}.post_feedforward_layernorm.weight")),
        _ => None,
    }
}

/// GGUF → HF tensor-name mapping. Behaviour-identical to the
/// pre-16.1 single function: try the architecture-agnostic common
/// rules first, then fall back to the family-specific extras for
/// `phi3` / `gemma2`. (Phase 16.2 moves the family dispatch behind
/// the adapter; this composing entry point is kept until 16.3
/// rewires the caller.)
pub fn gguf_to_hf_name(gguf_name: &str, arch: &str) -> Option<String> {
    if let Some(hf) = gguf_to_hf_name_common(gguf_name) {
        return Some(hf);
    }
    match arch {
        "phi3" => phi3_gguf_extra(gguf_name),
        "gemma2" => gemma2_gguf_extra(gguf_name),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_top_level_names() {
        assert_eq!(
            gguf_to_hf_name("token_embd.weight", "llama").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            gguf_to_hf_name("output_norm.weight", "llama").as_deref(),
            Some("model.norm.weight")
        );
        assert_eq!(
            gguf_to_hf_name("output.weight", "llama").as_deref(),
            Some("lm_head.weight")
        );
    }

    #[test]
    fn maps_llama_block_names() {
        let cases = [
            (
                "blk.7.attn_norm.weight",
                "model.layers.7.input_layernorm.weight",
            ),
            (
                "blk.7.attn_q.weight",
                "model.layers.7.self_attn.q_proj.weight",
            ),
            (
                "blk.7.attn_k.weight",
                "model.layers.7.self_attn.k_proj.weight",
            ),
            (
                "blk.7.attn_v.weight",
                "model.layers.7.self_attn.v_proj.weight",
            ),
            (
                "blk.7.attn_output.weight",
                "model.layers.7.self_attn.o_proj.weight",
            ),
            (
                "blk.7.ffn_norm.weight",
                "model.layers.7.post_attention_layernorm.weight",
            ),
            (
                "blk.7.ffn_gate.weight",
                "model.layers.7.mlp.gate_proj.weight",
            ),
            ("blk.7.ffn_up.weight", "model.layers.7.mlp.up_proj.weight"),
            (
                "blk.7.ffn_down.weight",
                "model.layers.7.mlp.down_proj.weight",
            ),
        ];
        for (gguf, hf) in cases {
            assert_eq!(gguf_to_hf_name(gguf, "llama").as_deref(), Some(hf));
        }
    }

    #[test]
    fn maps_architecture_specific_names() {
        assert_eq!(
            gguf_to_hf_name("blk.3.attn_qkv.weight", "phi3").as_deref(),
            Some("model.layers.3.self_attn.qkv_proj.weight")
        );
        assert_eq!(
            gguf_to_hf_name("blk.3.attn_post_norm.weight", "gemma2").as_deref(),
            Some("model.layers.3.post_attention_layernorm.weight")
        );
        assert_eq!(
            gguf_to_hf_name("blk.3.ffn_post_norm.weight", "gemma2").as_deref(),
            Some("model.layers.3.post_feedforward_layernorm.weight")
        );
    }

    #[test]
    fn rejects_unknown_or_malformed_names() {
        assert_eq!(gguf_to_hf_name("general.foo", "llama"), None);
        assert_eq!(gguf_to_hf_name("blk.x.attn_q.weight", "llama"), None);
        assert_eq!(gguf_to_hf_name("blk.0.unknown.weight", "llama"), None);
        assert_eq!(gguf_to_hf_name("blk.0.attn_qkv.weight", "llama"), None);
    }

    // ----- Phase 16.1: the split functions, exercised directly -----

    #[test]
    fn common_covers_top_level_and_llama_block_only() {
        assert_eq!(
            gguf_to_hf_name_common("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            gguf_to_hf_name_common("blk.7.attn_q.weight").as_deref(),
            Some("model.layers.7.self_attn.q_proj.weight")
        );
        // Family-specific suffixes are NOT common.
        assert_eq!(gguf_to_hf_name_common("blk.3.attn_qkv.weight"), None);
        assert_eq!(gguf_to_hf_name_common("blk.3.ffn_post_norm.weight"), None);
        // Malformed still rejected.
        assert_eq!(gguf_to_hf_name_common("blk.x.attn_q.weight"), None);
    }

    #[test]
    fn phi3_extra_only_handles_fused_qkv() {
        assert_eq!(
            phi3_gguf_extra("blk.3.attn_qkv.weight").as_deref(),
            Some("model.layers.3.self_attn.qkv_proj.weight")
        );
        assert_eq!(phi3_gguf_extra("blk.3.attn_q.weight"), None);
        assert_eq!(phi3_gguf_extra("token_embd.weight"), None);
    }

    #[test]
    fn gemma2_extra_only_handles_post_norms() {
        assert_eq!(
            gemma2_gguf_extra("blk.3.attn_post_norm.weight").as_deref(),
            Some("model.layers.3.post_attention_layernorm.weight")
        );
        assert_eq!(
            gemma2_gguf_extra("blk.3.post_attention_norm.weight").as_deref(),
            Some("model.layers.3.post_attention_layernorm.weight")
        );
        assert_eq!(
            gemma2_gguf_extra("blk.3.ffn_post_norm.weight").as_deref(),
            Some("model.layers.3.post_feedforward_layernorm.weight")
        );
        assert_eq!(gemma2_gguf_extra("blk.3.attn_q.weight"), None);
    }

    /// The composing entry point must remain byte-identical to the
    /// pre-16.1 behaviour: common first, then family fallback only
    /// for the matching arch.
    #[test]
    fn composing_fn_is_behaviour_identical() {
        assert_eq!(
            gguf_to_hf_name("blk.3.attn_qkv.weight", "phi3").as_deref(),
            Some("model.layers.3.self_attn.qkv_proj.weight")
        );
        // phi3 suffix under llama/gemma2 arch → None (unchanged).
        assert_eq!(gguf_to_hf_name("blk.3.attn_qkv.weight", "llama"), None);
        assert_eq!(gguf_to_hf_name("blk.3.attn_qkv.weight", "gemma2"), None);
        // gemma2 suffix only under gemma2 arch.
        assert_eq!(
            gguf_to_hf_name("blk.3.ffn_post_norm.weight", "gemma2").as_deref(),
            Some("model.layers.3.post_feedforward_layernorm.weight")
        );
        assert_eq!(gguf_to_hf_name("blk.3.ffn_post_norm.weight", "phi3"), None);
        // Common names resolve under any arch.
        assert_eq!(
            gguf_to_hf_name("blk.7.ffn_down.weight", "phi3").as_deref(),
            Some("model.layers.7.mlp.down_proj.weight")
        );
    }
}
