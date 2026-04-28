//! Regression test for the tied word embeddings code path (Phase A.1).
//!
//! Forces `tie_word_embeddings = true` on TinyLlama (which is normally
//! untied) and verifies that the tied path produces uniform behavior
//! across all 4 positions. This protects against future changes that
//! might break the tied path while still keeping the untied path
//! working.
//!
//! Originally introduced as Investigation D-fast during M4.6 SmolLM2
//! numerical validation, to rule out the tied path as the systemic
//! source of pos-0 drift. The tied path was confirmed structurally
//! correct (uniform per-position magnitudes); the SmolLM2 drift was
//! later traced to PyTorch BF16 precision rather than any Atenia bug
//! (see ADR-004). The test is retained as ongoing regression coverage
//! for the tied path itself.
//!
//! Note: with tied forced, Atenia's lm_head will be `embed_tokens.T`,
//! which is NOT equal to TinyLlama's actual `lm_head.weight`. The
//! resulting logits are therefore mathematically different from the
//! M4.5-d.1 baseline. We do NOT compare against PyTorch here; we only
//! verify pos-0 vs pos-1-3 magnitude distribution within the same
//! Atenia run.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::llama::{
    build_llama, llama_weight_mapper, LlamaConfig, LlamaRuntime,
};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::env;
use std::path::Path;

/// TinyLlama 1.1B Chat config WITH `tie_word_embeddings` flipped to
/// `true`. Everything else identical to the official checkpoint.
const EMBEDDED_TINYLLAMA_TIED_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5632,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 22,
  "num_key_value_heads": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.35.0",
  "use_cache": true,
  "vocab_size": 32000
}"#;

#[test]
#[ignore = "requires TINYLLAMA_SAFETENSORS_PATH"]
fn tinyllama_tied_path_remains_uniform_across_positions() {
    println!("\n=== TinyLlama tied-path regression: per-position uniformity check ===\n");

    let path = env::var("TINYLLAMA_SAFETENSORS_PATH")
        .expect("Set TINYLLAMA_SAFETENSORS_PATH");

    let config = LlamaConfig::from_json_str(EMBEDDED_TINYLLAMA_TIED_CONFIG)
        .expect("Failed to parse config with tied=true");
    assert!(config.tie_word_embeddings, "Config must have tied=true");
    println!(
        "Config: vocab={} hidden={} layers={} q_heads={} kv_heads={} tied={}",
        config.vocab_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.tie_word_embeddings,
    );

    let runtime = LlamaRuntime { batch: 1, seq: 4 };
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();

    // 1 (embed) + 22 × 9 (per layer) + 1 (final norm) = 200 — no lm_head.
    assert_eq!(
        handles.param_ids.len(),
        200,
        "Expected 200 params (no lm_head when tied), got {}",
        handles.param_ids.len()
    );
    assert!(
        !handles.param_names.iter().any(|n| n == "lm_head.weight"),
        "lm_head.weight must NOT be registered when tied"
    );

    println!("\nLoading weights...");
    let load_start = std::time::Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    let mapper = llama_weight_mapper(&config, &handles.param_names, &handles.param_ids)
        .expect("build mapper");
    let report = mapper.load_into(&mut graph, &reader).expect("load_into");
    drop(reader);
    println!(
        "Loaded {} tensors, skipped {}, missing {} in {:.2}s",
        report.loaded,
        report.skipped.len(),
        report.missing.len(),
        load_start.elapsed().as_secs_f32()
    );
    // Expect: 200 loaded (the params we registered) and exactly 1
    // skipped (lm_head.weight, which exists in the file but the
    // tied-path graph does not need).
    assert_eq!(report.loaded, 200, "expected 200 loaded");
    assert_eq!(report.skipped.len(), 1, "expected 1 skipped (lm_head.weight)");
    assert!(
        report.skipped.contains(&"lm_head.weight".to_string()),
        "Expected lm_head.weight in skipped, got {:?}",
        report.skipped
    );
    assert!(report.missing.is_empty(), "expected no missing");

    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 100.0, 200.0, 300.0]);
    println!("\nRunning forward...");
    let fwd_start = std::time::Instant::now();
    let outputs = graph.execute(vec![tokens]);
    println!("Forward: {:.2}s", fwd_start.elapsed().as_secs_f32());

    let logits = &outputs[0];
    assert_eq!(logits.shape, vec![1, 4, 32_000]);
    let slice = logits.as_cpu_slice();

    let finite = slice.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite, slice.len(), "non-finite logits detected");

    let vocab = 32_000_usize;
    println!("\n=== Per-position magnitude distribution ===");
    let mut per_pos: Vec<(usize, f32, f32, usize)> = Vec::new();
    for pos in 0..4 {
        let s = pos * vocab;
        let e = s + vocab;
        let row = &slice[s..e];
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let mean_abs: f32 = row.iter().map(|v| v.abs()).sum::<f32>() / vocab as f32;
        let count_over_10 = row.iter().filter(|v| v.abs() > 10.0).count();
        println!(
            "  Pos {}: max_abs={:.4}  mean_abs={:.4}  count(|v|>10)={}",
            pos, max_abs, mean_abs, count_over_10
        );
        per_pos.push((pos, max_abs, mean_abs, count_over_10));
    }

    // Quick disparity check: ratio of pos 0 max_abs to median of pos 1-3.
    let mut other_means: Vec<f32> = per_pos[1..].iter().map(|(_, _, m, _)| *m).collect();
    other_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_mean = other_means[other_means.len() / 2];
    let pos0_mean = per_pos[0].2;
    let mean_ratio = if median_mean > 0.0 {
        pos0_mean / median_mean
    } else {
        f32::NAN
    };
    println!(
        "\nMean-abs ratio  pos0/median(pos1-3)  =  {:.3}",
        mean_ratio
    );

    let pos0_max = per_pos[0].1;
    let other_maxes: Vec<f32> = per_pos[1..].iter().map(|(_, m, _, _)| *m).collect();
    let max_max_other = other_maxes.iter().fold(0.0_f32, |a, &b| a.max(b));
    let max_ratio = if max_max_other > 0.0 {
        pos0_max / max_max_other
    } else {
        f32::NAN
    };
    println!(
        "Max-abs ratio   pos0/max(pos1-3)     =  {:.3}",
        max_ratio
    );

    // Regression threshold: per-position magnitudes must remain uniform.
    // Empirically the tied path produces ratios near 1.0 across positions;
    // a regression that introduces position-specific behavior in this
    // path would push the ratio above 2×.
    assert!(
        mean_ratio < 2.0,
        "Tied path regression: mean-abs ratio pos0/median(pos1-3) = {:.3} \
         (expected < 2.0). Per-position behavior of the tied LM head should \
         be uniform.",
        mean_ratio
    );
    assert!(
        max_ratio < 2.0,
        "Tied path regression: max-abs ratio pos0/max(pos1-3) = {:.3} \
         (expected < 2.0).",
        max_ratio
    );

    println!("\nPASSED: tied path remains uniform across positions.");
}
