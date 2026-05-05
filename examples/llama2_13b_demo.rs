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
use atenia_engine::cuda::disk_prefetch::disk_prefetch_hits;
use atenia_engine::cuda::matmul::{
    disk_streamed_matmul_count, vram_bf16_matmul_count,
};
use atenia_engine::gpu::dispatch::hooks::{
    gpu_matmul_legacy_count, gpu_matmul_resident_count, gpu_matmul_roundtrip_count,
};
use atenia_engine::gpu::safety::resource_check::{
    probe_free_ram_bytes, probe_free_vram_bytes, probe_total_ram_bytes,
};
use atenia_engine::gpu::tier_plan::{plan as tier_plan_fn, TensorMeta, TierPlanInput};
use atenia_engine::nn::llama::{
    build_llama, build_llama_with_store, llama_weight_mapper, LlamaConfig,
    LlamaRuntime,
};
use atenia_engine::tensor::tensor::Tensor;
use atenia_engine::tensor::DType;
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

    // ---- Build scratch graph ----
    //
    // The tier-aware loader needs a `Graph` with the full
    // parameter schema to validate shapes / dtypes against
    // the safetensors header. After the load it returns a
    // populated `WeightStore` and the scratch graph is
    // discarded — for VRAM/Disk-tier weights, the loader
    // never writes the parameter slots in the graph it
    // received (only Ram-tier weights are extracted via
    // `finalize_ram_extract`). The forward graph is rebuilt
    // below with `build_llama_with_store` so every parameter
    // node is connected to the store via
    // `register_param_from_store`. This matches the
    // canonical pattern in `LlamaPipeline::load` /
    // `generate_greedy`.
    println!("\n[1/4] Building graph ...");
    let build_start = Instant::now();
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &cfg, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut scratch_graph = gb.build();
    println!(
        "    Graph built in {:.2}s ({} parameter nodes)",
        build_start.elapsed().as_secs_f32(),
        handles.param_ids.len(),
    );

    // ---- Load weights (tier-aware) ----
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

    // Tier-aware load (M6 + M7 + M8 + M8.7) — same path
    // `LlamaPipeline::load` takes by default. Probes free
    // RAM/VRAM, builds a `TierPlan` from the safetensors
    // header metadata, and routes each parameter to its
    // assigned tier (`Vram` / `Ram` / `Disk`) via
    // `load_into_with_residency_plan`.
    let metas: Vec<TensorMeta> = sharded
        .collect_tensor_metas()
        .expect("collect_tensor_metas");
    let model_total_bytes: u64 = metas
        .iter()
        .map(|m| {
            let numel = m.shape.iter().product::<usize>() as u64;
            numel * (m.dtype.size_in_bytes() as u64)
        })
        .sum();
    let kernel_dtype = if std::env::var("ATENIA_M8_BF16_KERNEL")
        .as_deref()
        == Ok("1")
    {
        DType::BF16
    } else {
        DType::F32
    };
    mapper.set_bf16_kernel_active(Some(kernel_dtype == DType::BF16));
    let plan_input = TierPlanInput {
        tensors: metas,
        free_vram_bytes: probe_free_vram_bytes(),
        free_ram_bytes: probe_free_ram_bytes(),
        model_total_bytes,
        total_ram_bytes: probe_total_ram_bytes(),
        kernel_dtype,
    };
    let plan = tier_plan_fn(&plan_input);
    eprintln!(
        "    Tier plan: vram={} tensors ({:.2} GiB), ram={} tensors, disk={} tensors",
        plan.vram_count(),
        plan.vram_bytes_assigned as f64 / 1024.0_f64.powi(3),
        plan.ram_count(),
        plan.disk_count(),
    );
    let (store, report) = sharded
        .load_into_with_residency_plan(
            &mut scratch_graph,
            &mapper,
            &plan,
            &handles.param_ids,
            &handles.param_names,
        )
        .expect("load_into_with_residency_plan");
    drop(scratch_graph);
    let load_secs = load_start.elapsed().as_secs_f32();
    println!(
        "    Loaded {} tensors in {:.2}s (~{:.0} MB/s)",
        report.loaded,
        load_secs,
        26_000.0 / load_secs.max(0.01),
    );

    // ---- Rebuild graph wired to the store ----
    //
    // `build_llama_with_store` creates a fresh `GraphBuilder`
    // backed by the `WeightStore`: every parameter node is
    // attached to its `SharedParam::{Cuda, Disk, Bf16, F32}`
    // via `register_param_from_store`. The dispatch hooks
    // (`gpu/dispatch/hooks.rs::try_gpu_matmul`,
    // `cuda::matmul::cuda_matmul_disk_streamed_bf16`) read
    // the `SharedParam` variant directly off the parameter's
    // storage at execute time.
    let mut gb2 = GraphBuilder::new();
    let token_input_id_2 = gb2.input();
    let handles2 = build_llama_with_store(
        &mut gb2,
        &cfg,
        &runtime,
        token_input_id_2,
        &store,
        None,
    )
    .expect("build_llama_with_store");
    let _ = gb2.output(handles2.logits_id);
    let mut graph = gb2.build();

    // ---- Snapshot GPU MatMul counters ----
    let resident_before = gpu_matmul_resident_count();
    let roundtrip_before = gpu_matmul_roundtrip_count();
    let legacy_before = gpu_matmul_legacy_count();
    let bf16_before = vram_bf16_matmul_count();
    let disk_streamed_before = disk_streamed_matmul_count();
    let disk_prefetch_before = disk_prefetch_hits();

    // ---- Forward ----
    println!(
        "\n[3/4] Running forward at seq={}, tokens={:?} ...",
        runtime.seq, token_pattern,
    );
    let tokens = Tensor::new_cpu(vec![1, runtime.seq], token_pattern);
    let fwd_start = Instant::now();
    let outputs = graph.execute_inference(vec![tokens]);
    let fwd_secs = fwd_start.elapsed().as_secs_f32();

    let resident_after = gpu_matmul_resident_count();
    let roundtrip_after = gpu_matmul_roundtrip_count();
    let legacy_after = gpu_matmul_legacy_count();
    let bf16_after = vram_bf16_matmul_count();
    let disk_streamed_after = disk_streamed_matmul_count();
    let disk_prefetch_after = disk_prefetch_hits();
    let resident_delta = resident_after - resident_before;
    let roundtrip_delta = roundtrip_after - roundtrip_before;
    let legacy_delta = legacy_after - legacy_before;
    let bf16_delta = bf16_after - bf16_before;
    let disk_streamed_delta = disk_streamed_after - disk_streamed_before;
    let disk_prefetch_delta = disk_prefetch_after - disk_prefetch_before;

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
        "    BF16-resident matmuls (M8.4c): {}",
        bf16_delta,
    );
    println!(
        "    Disk-streamed matmuls (M8.7.0): {}",
        disk_streamed_delta,
    );
    println!(
        "    Disk prefetch hits (M8.7.1.a): {}",
        disk_prefetch_delta,
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
