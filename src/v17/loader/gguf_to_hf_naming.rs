pub fn gguf_to_hf_name(gguf_name: &str, arch: &str) -> Option<String> {
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        "rope_freqs.weight" => return Some("rope_freqs".to_string()),
        _ => {}
    }

    let rest = gguf_name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    if layer.is_empty() || !layer.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }

    let prefix = format!("model.layers.{layer}");
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
        "attn_qkv.weight" if arch == "phi3" => Some(format!("{prefix}.self_attn.qkv_proj.weight")),
        "attn_post_norm.weight" if arch == "gemma2" => {
            Some(format!("{prefix}.post_attention_layernorm.weight"))
        }
        "post_attention_norm.weight" if arch == "gemma2" => {
            Some(format!("{prefix}.post_attention_layernorm.weight"))
        }
        "ffn_post_norm.weight" if arch == "gemma2" => {
            Some(format!("{prefix}.post_feedforward_layernorm.weight"))
        }
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
}
