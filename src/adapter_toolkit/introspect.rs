//! **Adapter Toolkit v2 — Part 7: introspection / debug.**
//!
//! Human-readable rendering of a [`GeneratedAdapter`]: the resolved
//! family and architecture, the v1 adapter it delegates to, the
//! v1 capability flags, the normalised [`FeatureSet`], every
//! per-checkpoint override, and a sample of the GGUF→HF tensor-name
//! mapping the adapter exposes.
//!
//! Two render depths share one renderer: `summary` (the `atenia
//! load` view) and `verbose` (the `atenia debug` view, which adds
//! capabilities and the tensor-name sample). Rendering is pure —
//! it reads the adapter, never builds a graph, never touches a
//! model file.

use crate::model_adapters::{GgufNameMapper, ModelAdapter};

use super::generator::GeneratedAdapter;
use super::spec::{KvHeadsResolved, ResolvedAdapterSpec};

/// Representative GGUF tensor names probed for the verbose
/// mapping sample — one per major weight class plus the two
/// fused-tensor names only Phi-3 resolves.
const SAMPLE_GGUF_NAMES: &[&str] = &[
    "token_embd.weight",
    "output_norm.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_qkv.weight",
    "blk.0.ffn_down.weight",
    "blk.0.ffn_up.weight",
];

/// Render a [`GeneratedAdapter`] as a multi-line report.
///
/// - `verbose = false` → the `atenia load` summary.
/// - `verbose = true`  → the `atenia debug` view (adds the v1
///   capability flags and the GGUF→HF tensor-name sample).
pub fn describe_adapter(adapter: &GeneratedAdapter, verbose: bool) -> String {
    let spec = adapter.spec();
    let mut out = String::new();

    out.push_str("Adapter Toolkit v2 — generated adapter\n");
    out.push_str("======================================\n");
    push_spec(&mut out, spec, adapter.base_id());

    if verbose {
        push_capabilities(&mut out, adapter);
        push_tensor_sample(&mut out, adapter);
    }

    out
}

/// Render a bare [`ResolvedAdapterSpec`] (no live adapter) — used by
/// `atenia inspect` to preview the spec it generated.
pub fn describe_spec(spec: &ResolvedAdapterSpec) -> String {
    let mut out = String::new();
    out.push_str("Adapter Toolkit v2 — resolved spec\n");
    out.push_str("==================================\n");
    push_spec(&mut out, spec, "<not bound>");
    out
}

fn push_spec(out: &mut String, spec: &ResolvedAdapterSpec, base_id: &str) {
    out.push_str(&format!("  family          : {:?}\n", spec.family));
    out.push_str(&format!("  architecture    : {}\n", spec.architecture));
    out.push_str(&format!("  model_type      : {}\n", spec.model_type));
    out.push_str(&format!("  v1 base adapter : {base_id}\n"));

    let f = &spec.features;
    // These are *declared* features: the DSL's `config` / `weights`
    // / `attention` sections are expected constraints, validated
    // for consistency and used by introspection. They do NOT mutate
    // the model's LlamaConfig — config.json / GGUF metadata remain
    // authoritative. The label says so explicitly so the report
    // never implies the YAML controls the runtime config.
    out.push_str("  declared features (validated, not applied — config.json is authoritative):\n");
    out.push_str(&format!("    rope          : {}\n", f.rope.label()));
    if let Some(prf) = f.partial_rotary_factor {
        out.push_str(&format!("    partial_rotary: {prf}\n"));
    }
    out.push_str(&format!("    attention     : {}\n", f.attention.label()));
    let kv = match f.kv_heads {
        KvHeadsResolved::Auto => "auto (from config.json)".to_string(),
        KvHeadsResolved::Count(n) => n.to_string(),
    };
    out.push_str(&format!("    kv_heads      : {kv}\n"));
    out.push_str(&format!("    fused_qkv     : {}\n", f.fused_qkv));
    out.push_str(&format!("    fused_mlp     : {}\n", f.fused_mlp));
    if let Some(s) = &f.split_strategy {
        out.push_str(&format!("    split_strategy: {s}\n"));
    }

    out.push_str("  tokenizer:\n");
    match &spec.tokenizer.eos_tokens {
        Some(e) => out.push_str(&format!("    eos_tokens    : {e:?}\n")),
        None => out.push_str("    eos_tokens    : (from config.json / GGUF)\n"),
    }
    if spec.tokenizer.turn_terminators.is_empty() {
        out.push_str("    turn_terminators: (none declared)\n");
    } else {
        out.push_str(&format!(
            "    turn_terminators: {:?}\n",
            spec.tokenizer.turn_terminators
        ));
    }

    if spec.overrides.is_empty() {
        out.push_str("  overrides       : (none)\n");
    } else {
        out.push_str(&format!("  overrides       : {} declared\n", spec.overrides.len()));
        for ov in &spec.overrides {
            let eos = ov
                .tokenizer
                .eos_tokens
                .as_ref()
                .map(|e| format!("{e:?}"))
                .unwrap_or_else(|| "(inherits base)".to_string());
            out.push_str(&format!("    - {:<20} eos_tokens={eos}\n", ov.label));
        }
    }
}

fn push_capabilities(out: &mut String, adapter: &GeneratedAdapter) {
    let c = adapter.capabilities();
    out.push_str("  v1 capabilities (delegated):\n");
    out.push_str(&format!("    hf_safetensors          : {}\n", c.hf_safetensors));
    out.push_str(&format!("    gguf                    : {}\n", c.gguf));
    out.push_str(&format!(
        "    store_backed_generation : {}\n",
        c.store_backed_generation
    ));
    out.push_str(&format!(
        "    fused_qkv_weight_mapping: {}\n",
        c.fused_qkv_weight_mapping
    ));
    out.push_str(&format!(
        "    fused_gate_up_mapping   : {}\n",
        c.fused_gate_up_weight_mapping
    ));
    out.push_str(&format!("    gemma2_softcaps         : {}\n", c.gemma2_softcaps));
}

fn push_tensor_sample(out: &mut String, adapter: &GeneratedAdapter) {
    out.push_str("  GGUF -> HF tensor-name sample (v1 mapping, delegated):\n");
    for name in SAMPLE_GGUF_NAMES {
        match adapter.gguf_to_hf_name(name) {
            Some(hf) => out.push_str(&format!("    {name:<24} -> {hf}\n")),
            None => out.push_str(&format!("    {name:<24} -> (not mapped by this family)\n")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_toolkit::dsl::AdapterDsl;

    fn generated(text: &str) -> GeneratedAdapter {
        let dsl = AdapterDsl::from_str(text, true).expect("dsl parses");
        let spec = ResolvedAdapterSpec::resolve(&dsl).expect("spec resolves");
        GeneratedAdapter::from_spec(spec).expect("adapter generates")
    }

    #[test]
    fn summary_reports_family_and_base_adapter() {
        let report = describe_adapter(&generated("family: llama\n"), false);
        assert!(report.contains("family"));
        assert!(report.contains("Llama"));
        assert!(report.contains("v1 base adapter : llama"));
    }

    #[test]
    fn summary_omits_capabilities_and_tensor_sample() {
        let report = describe_adapter(&generated("family: llama\n"), false);
        assert!(!report.contains("v1 capabilities"));
        assert!(!report.contains("tensor-name sample"));
    }

    #[test]
    fn verbose_includes_capabilities_and_tensor_sample() {
        let report = describe_adapter(&generated("family: phi\n"), true);
        assert!(report.contains("v1 capabilities"));
        assert!(report.contains("tensor-name sample"));
        // Phi-3 fused QKV must resolve in the sample.
        assert!(report.contains("self_attn.qkv_proj.weight"));
        assert!(report.contains("fused_qkv_weight_mapping: true"));
    }

    #[test]
    fn verbose_llama_shows_unmapped_fused_qkv() {
        let report = describe_adapter(&generated("family: llama\n"), true);
        // The common llama table does not map attn_qkv.
        assert!(report.contains("attn_qkv.weight") && report.contains("(not mapped"));
    }

    #[test]
    fn report_lists_overrides() {
        let report = describe_adapter(
            &generated(
                "family: qwen\n\
                 overrides:\n  deepseek-distill:\n    tokenizer:\n      eos_tokens: [1, 106]\n",
            ),
            false,
        );
        assert!(report.contains("overrides"));
        assert!(report.contains("deepseek-distill"));
        assert!(report.contains("[1, 106]"));
    }

    #[test]
    fn describe_spec_renders_without_a_live_adapter() {
        let dsl = AdapterDsl::from_str("family: gemma3\n", true).unwrap();
        let spec = ResolvedAdapterSpec::resolve(&dsl).unwrap();
        let report = describe_spec(&spec);
        assert!(report.contains("Gemma3"));
        assert!(report.contains("<not bound>"));
    }
}
