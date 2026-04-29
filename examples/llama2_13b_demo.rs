//! M4.7.6.d — Llama 2 13B Chat killer-demo runner.
//!
//! Mode A only in this commit. Modes B and C land in M4.7.6.e.
//!
//! Usage:
//! ```powershell
//! cargo run --release --example llama2_13b_demo
//! ```
//!
//! Builds the standard Llama 2 13B Chat graph at seq=4, loads
//! 363 parameters across 3 shards via `ShardedSafetensorsReader`
//! with BF16 storage active (M4.7.2), runs the forward with the
//! M4.7.6.c GPU MatMul wiring active, and prints the metrics
//! used by the demo's M4.7.6.e tri-mode comparison:
//!
//!   - Wall-clock load time + forward time.
//!   - Per-position argmax (token id).
//!   - GPU MatMul counter delta (proves M4.7.6.c wiring fired).
//!   - Logit sanity (max |v|, mean |v|, NaN / Inf count).
//!
//! Mode A constraint: clean RAM, no spill, no LRU policy. The
//! example does not attach a `ReactiveExecutionContext`, so the
//! M3-e reaction loop does not fire. M4.7.6.e adds Mode B (LRU
//! spill under simulated foreground pressure) and Mode C (force
//! spill via `deep_degrade_with_lru` pre-forward); both reuse
//! this loader / forward pipeline.
//!
//! Hardware prerequisite: ~28 GB RAM available (26 GB params +
//! ~2 GB activations / overhead). Dev-box default is 32 GB; close
//! Chrome / VS Code before running for the cleanest baseline.

use std::time::Instant;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::gpu::dispatch::hooks::{
    gpu_matmul_legacy_count, gpu_matmul_resident_count, gpu_matmul_roundtrip_count,
};
use atenia_engine::nn::llama::{
    build_llama, llama_weight_mapper, LlamaConfig, LlamaRuntime,
};
use atenia_engine::tensor::tensor::Tensor;
use atenia_engine::v17::loader::sharded_reader::ShardedSafetensorsReader;

/// Llama 2 13B Chat lives on the **internal NVMe (D:)** for the
/// demo. F: USB HDD throughput (~7.5 MB/s) is too slow to be
/// usable; the demo's binding constraint is RAM↔disk traffic and
/// the model path must be on the fast tier. Project root stays
/// on F: per dev-box convention; only the runtime data tier
/// moves to D:. Override via `ATENIA_LLAMA2_13B_DIR` env var.
const DEFAULT_MODEL_DIR: &str = "D:/Atenia/models/llama-2-13b-chat";

fn resolve_model_dir() -> std::path::PathBuf {
    match std::env::var("ATENIA_LLAMA2_13B_DIR") {
        Ok(v) => std::path::PathBuf::from(v),
        Err(_) => std::path::PathBuf::from(DEFAULT_MODEL_DIR),
    }
}

fn main() {
    println!("\n=== Atenia v20 Killer Demo — Llama 2 13B Chat (Mode A) ===\n");

    let model_dir = resolve_model_dir();
    if !model_dir.exists() {
        eprintln!(
            "ERROR: model directory not found: {}\n\
             The demo expects Llama 2 13B Chat on D: (NVMe data tier).\n\
             Download to project root, then copy to D::\n  \
             huggingface-cli download meta-llama/Llama-2-13b-chat-hf \\\n    \
             --local-dir <project_root>/models/llama-2-13b-chat \\\n    \
             --include '*.safetensors' '*.json' 'tokenizer*'\n  \
             cp -r <project_root>/models/llama-2-13b-chat D:/Atenia/models/\n\
             Or override path via env: ATENIA_LLAMA2_13B_DIR=<your_dir>",
            model_dir.display(),
        );
        std::process::exit(1);
    }
    let model_dir = model_dir.as_path();

    let cfg = LlamaConfig::from_json_file(&model_dir.join("config.json"))
        .expect("config.json must parse");
    println!(
        "Architecture: {} layers × hidden {} × intermediate {} ({} attention heads, \
         {} kv heads), vocab {}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size,
    );

    let runtime = LlamaRuntime { batch: 1, seq: 4 };
    let token_pattern: Vec<f32> = vec![1.0, 100.0, 200.0, 300.0];

    // ---- Build graph ----
    println!("\n[1/4] Building graph ...");
    let build_start = Instant::now();
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &cfg, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    println!(
        "    Graph built in {:.2}s ({} parameter nodes)",
        build_start.elapsed().as_secs_f32(),
        handles.param_ids.len(),
    );

    // ---- Load weights ----
    println!(
        "\n[2/4] Loading weights from {} (BF16 storage active) ...",
        model_dir.display()
    );
    let load_start = Instant::now();
    let sharded = ShardedSafetensorsReader::open(&model_dir.join("model.safetensors.index.json"))
        .expect("open sharded reader");
    let mut mapper = llama_weight_mapper(&cfg, &handles.param_names, &handles.param_ids)
        .expect("llama weight mapper");
    mapper.set_store_params_as_bf16(true);
    let report = sharded.load_into(&mut graph, &mapper).expect("load");
    let load_secs = load_start.elapsed().as_secs_f32();
    println!(
        "    Loaded {} tensors in {:.2}s (~{:.0} MB/s)",
        report.loaded,
        load_secs,
        26_000.0 / load_secs.max(0.01),
    );

    // ---- Snapshot GPU MatMul counters ----
    let resident_before = gpu_matmul_resident_count();
    let roundtrip_before = gpu_matmul_roundtrip_count();
    let legacy_before = gpu_matmul_legacy_count();

    // ---- Forward ----
    println!(
        "\n[3/4] Running forward at seq={}, tokens={:?} ...",
        runtime.seq, token_pattern,
    );
    let tokens = Tensor::new_cpu(vec![1, runtime.seq], token_pattern);
    let fwd_start = Instant::now();
    let outputs = graph.execute(vec![tokens]);
    let fwd_secs = fwd_start.elapsed().as_secs_f32();

    let resident_after = gpu_matmul_resident_count();
    let roundtrip_after = gpu_matmul_roundtrip_count();
    let legacy_after = gpu_matmul_legacy_count();
    let resident_delta = resident_after - resident_before;
    let roundtrip_delta = roundtrip_after - roundtrip_before;
    let legacy_delta = legacy_after - legacy_before;

    let logits = &outputs[0];
    assert_eq!(logits.shape, vec![1, runtime.seq, cfg.vocab_size]);
    let slice = logits.as_cpu_slice();

    // ---- Metrics ----
    let finite_count = slice.iter().filter(|v| v.is_finite()).count();
    let max_abs = slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let mean_abs: f32 = slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len() as f32;

    println!("    Forward: {:.2}s", fwd_secs);
    println!(
        "    GPU MatMul invocations: resident={}, roundtrip={}, legacy={}, total={}",
        resident_delta,
        roundtrip_delta,
        legacy_delta,
        resident_delta + roundtrip_delta + legacy_delta,
    );
    println!(
        "    Logit stats: max |v|={:.4}  mean |v|={:.4}  finite={}/{}",
        max_abs,
        mean_abs,
        finite_count,
        slice.len(),
    );

    // ---- Per-position argmax ----
    println!("\n[4/4] Per-position argmax:");
    for pos in 0..runtime.seq {
        let s = pos * cfg.vocab_size;
        let e = s + cfg.vocab_size;
        let row = &slice[s..e];
        let (id, val) = row
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        println!("    Pos {}: argmax id={:>5}  logit={:.4}", pos, id, val);
    }

    println!(
        "\n=== Mode A complete. Wall-clock totals: build {:.1}s | load {:.1}s | forward {:.1}s ===\n",
        build_start.elapsed().as_secs_f32() - load_secs - fwd_secs,
        load_secs,
        fwd_secs,
    );
}
