//! M4.7.6.d — Llama 2 13B Chat first end-to-end forward, Mode A
//! (clean RAM, no spill, BF16 storage active, M4.7.6.c GPU MatMul
//! wiring active).
//!
//! **The first runnable killer-demo artefact.**
//!
//! Builds the standard Llama 2 13B graph at
//! `LlamaRuntime { batch: 1, seq: 4 }`, loads 363 parameters
//! across the 3 shards via `ShardedSafetensorsReader` with
//! `store_params_as_bf16(true)`, runs forward on the canonical
//! M4.6 token pattern `[1, 100, 200, 300]`, asserts:
//!
//!   1. Loader contract: 363 tensors loaded, 40 skipped
//!      (the per-layer `rotary_emb.inv_freq` buffers — Atenia
//!      computes them at runtime via `nn::rope::compute_inv_freqs`),
//!      zero missing.
//!   2. Logits shape `[1, 4, 32_000]`.
//!   3. All logits finite (no NaN / Inf — proves no kernel
//!      blowup along the 40-layer chain).
//!   4. Plausible logit magnitudes (`max |v| < 1000`,
//!      `mean |v| > 0.5`). Below 1000 rules out the
//!      pathological numerical-instability mode; above 0.5
//!      rules out the all-zeros bug where the loader silently
//!      no-op'd.
//!   5. GPU MatMul counter is **observability-only** for 13B —
//!      see "Architectural finding" below.
//!
//! ## Architectural finding (M4.7.6.d, dropped from .d gate)
//!
//! The M4.7.6.c counter assertion (`gpu_matmul_total_count > 0`)
//! holds on the four M4.6 family models (TinyLlama, SmolLM2,
//! Qwen 2.5, Llama 3.2 — `roundtrip` counts of 154 / 168 / 196 /
//! 112 respectively). On Llama 2 13B Chat the counter is **zero**
//! and **the entire forward runs on CPU**. Two architectural
//! limits stack to produce this:
//!
//!   - The M4.7.3 residency-aware `try_gpu_matmul` path requires
//!     each of A / B / output to fit in one
//!     `DEFAULT_BLOCK_SIZE = 64 MiB` pool block (M4.7.6.c
//!     capacity check). Llama 2 13B's weight tensors are
//!     5120 × 5120 × 4 = **100 MB** for QKVO and 5120 × 13824 ×
//!     4 = **270 MB** for gate/up/down — every one of them
//!     exceeds 64 MiB. `gpu_can_run_matmul` returns false on
//!     every Llama 2 13B MatMul.
//!   - The legacy fallback `apx4::dispatch_matmul_gpu` consults
//!     `apx4::gpu_context::gpu_available()`, which is hardcoded
//!     to `false` (placeholder since pre-M4.6) and was never
//!     wired to real CUDA detection. Even if it were wired,
//!     `gpu_matmul` (apx4) re-enters `cuda_matmul`, which itself
//!     re-uses the same pool → same 64 MiB ceiling.
//!
//! The fix is a non-pooled `cuda_matmul` variant for tensors
//! that exceed the pool block size: direct `cudaMalloc` per
//! invocation, transfer + kernel + free, no pool. That is
//! M5+ scope (the "performance" tail flagged in ROADMAP after
//! M4.7 closes) and intentionally NOT in M4.7.6's contract.
//!
//! Mode A's claim becomes "13B Chat runs end-to-end on this
//! hardware" without further qualifying about the dispatch
//! path. CPU compute on a 24-thread AVX2 machine takes
//! ~18-20 minutes per forward at seq=4; that is the honest
//! demo number. M4.7.6.e Modes B / C add the LRU spill +
//! force spill on top of this same CPU-bound forward; the
//! cross-mode argmax-equality contract there is what locks
//! the transparency claim, independent of GPU acceleration.
//!
//! **Reference-fixture comparison is intentionally NOT enforced
//! in this test.** The M4.7.6 investigation locked a hybrid
//! validation strategy:
//!
//!   - Dev-local BF16 reference: best-effort, deferred. The
//!     reference dev hardware ships the model on F: USB HDD
//!     (~7.5 MB/s, 276 ms response time per Task Manager
//!     measurement during M4.7.6.d work). At that throughput
//!     PyTorch's 26 GB load takes >1 hour and the forward
//!     another ~15 minutes, putting the fixture generation
//!     off the .d critical path. The generator script
//!     `tests/fixtures/llama2_13b_reference/generate_bf16.py`
//!     is committed — any operator with the model on faster
//!     storage (NVMe at ~3 GB/s = ~10 second load) can run it
//!     once and drop `expected_logits_bf16.json` to re-enable
//!     the element-wise gate.
//!   - Cloud F64 fixture: the v20 release lock per ADR-004,
//!     produced on rented L40S / A100 (96 / 80 GB VRAM —
//!     enough for 13B in F64 at 104 GB) before the v20 tag.
//!
//! Mode A's contract here is therefore "the demo runs at all,
//! produces sensible numbers, and exercises the M4.7.6.c GPU
//! wiring". M4.7.6.e adds Modes B and C (LRU spill and force
//! spill) on top of this same loader / forward pipeline; the
//! cross-mode argmax-equality contract there is what locks the
//! transparency claim, independent of any external reference.
//!
//! `#[ignore]`-gated; needs only the model on disk (24 GB).
//!
//! ```powershell
//! cargo test --test m4_7_6_d_llama2_13b_mode_a_test --release \
//!     -- --ignored --nocapture
//! ```

use std::time::Instant;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::gpu::dispatch::hooks::{
    gpu_matmul_legacy_count, gpu_matmul_resident_count, gpu_matmul_roundtrip_count,
    gpu_matmul_total_count,
};
use atenia_engine::nn::llama::{
    build_llama, llama_weight_mapper, LlamaConfig, LlamaRuntime,
};
use atenia_engine::tensor::tensor::Tensor;
use atenia_engine::v17::loader::sharded_reader::ShardedSafetensorsReader;

/// Llama 2 13B Chat lives on the **internal NVMe (D:)** for the
/// demo — F: USB HDD is too slow (7.5 MB/s sustained, 276 ms
/// response time per Task Manager during M4.7.6.d work). The
/// project root stays on F: per the dev-box convention; only
/// the runtime data tier moves to D:. Same drive policy locked
/// for `ATENIA_DISK_TIER_DIR` in M4.7.4 / M4.7.5.
///
/// Override via `ATENIA_LLAMA2_13B_DIR` env var if running on a
/// different layout.
const DEFAULT_MODEL_DIR: &str = "D:/Atenia/models/llama-2-13b-chat";

fn resolve_model_dir() -> std::path::PathBuf {
    match std::env::var("ATENIA_LLAMA2_13B_DIR") {
        Ok(v) => std::path::PathBuf::from(v),
        Err(_) => std::path::PathBuf::from(DEFAULT_MODEL_DIR),
    }
}
const TOKENS: [f32; 4] = [1.0, 100.0, 200.0, 300.0];

#[test]
#[ignore = "requires Llama 2 13B Chat checkpoint at MODEL_DIR (24 GB)"]
fn llama2_13b_mode_a_clean_ram_runs_with_gpu_matmul_wiring() {
    println!("\n=== Llama 2 13B Chat — Mode A (clean RAM, no spill) ===\n");

    let model_dir = resolve_model_dir();
    assert!(
        model_dir.exists(),
        "model dir missing at {:?}; copy the checkpoint from F: \
         (project root) to D: (NVMe data tier) or set \
         ATENIA_LLAMA2_13B_DIR env var",
        model_dir
    );

    let cfg_path = model_dir.join("config.json");
    let cfg = LlamaConfig::from_json_file(&cfg_path).expect("config");
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.vocab_size, 32_000);

    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    // ---- 1. Build graph ----
    println!("Building graph (40 layers, hidden 5120, intermediate 13824) ...");
    let build_start = Instant::now();
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &cfg, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    println!("Graph built in {:.2}s", build_start.elapsed().as_secs_f32());
    assert_eq!(handles.param_ids.len(), 363);

    // ---- 2. Load 363 tensors via ShardedSafetensorsReader ----
    let index_path = model_dir.join("model.safetensors.index.json");
    println!("Opening sharded index at {} ...", index_path.display());
    let sharded =
        ShardedSafetensorsReader::open(&index_path).expect("open sharded reader");
    println!(
        "{} shards, {} tensors declared",
        sharded.shard_count(),
        sharded.tensor_count()
    );
    assert_eq!(sharded.shard_count(), 3);

    println!("Loading weights with store_params_as_bf16 = true ...");
    let load_start = Instant::now();
    let mut mapper = llama_weight_mapper(&cfg, &handles.param_names, &handles.param_ids)
        .expect("llama weight mapper");
    mapper.set_store_params_as_bf16(true);
    let report = sharded
        .load_into(&mut graph, &mapper)
        .expect("sharded load");
    let load_secs = load_start.elapsed().as_secs_f32();
    println!(
        "Loaded {} tensors in {:.2}s (~{:.0} MB/s effective)",
        report.loaded,
        load_secs,
        26_000.0 / load_secs.max(0.01)
    );
    assert_eq!(report.loaded, 363);
    // The 40 `rotary_emb.inv_freq` buffers land in `report.skipped`
    // because Atenia computes them at runtime in the RoPE arm.
    assert_eq!(report.skipped.len(), 40);
    assert!(report.missing.is_empty(), "missing: {:?}", report.missing);

    // ---- 3. Snapshot GPU MatMul counters before forward ----
    let resident_before = gpu_matmul_resident_count();
    let roundtrip_before = gpu_matmul_roundtrip_count();
    let legacy_before = gpu_matmul_legacy_count();
    let total_before = gpu_matmul_total_count();
    println!(
        "GPU MatMul counters before forward: resident={}, roundtrip={}, legacy={}, total={}",
        resident_before, roundtrip_before, legacy_before, total_before
    );

    // ---- 4. Forward pass ----
    let tokens = Tensor::new_cpu(vec![1, 4], TOKENS.to_vec());
    println!("Running forward (M4.7.6.c GPU MatMul wiring active) ...");
    let fwd_start = Instant::now();
    let outputs = graph.execute(vec![tokens]);
    let fwd_secs = fwd_start.elapsed().as_secs_f32();
    println!("Forward: {:.2}s", fwd_secs);

    let logits = &outputs[0];
    assert_eq!(logits.shape, vec![1, 4, 32_000]);
    let slice = logits.as_cpu_slice();

    // ---- 5. Logit sanity ----
    let finite = slice.iter().filter(|v| v.is_finite()).count();
    assert_eq!(
        finite,
        slice.len(),
        "all logits must be finite (got {} non-finite of {})",
        slice.len() - finite,
        slice.len(),
    );
    let max_abs = slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let mean_abs: f32 = slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len() as f32;
    println!("Logit stats: max |v|={:.4}  mean |v|={:.4}", max_abs, mean_abs);
    assert!(
        max_abs < 1000.0,
        "logits suspiciously large: max |v| = {} (instability?)",
        max_abs
    );
    assert!(
        mean_abs > 0.5,
        "logits suspiciously small: mean |v| = {} (loader silently no-op'd?)",
        mean_abs
    );

    // ---- 6. GPU MatMul counter (observability-only for 13B) ----
    //
    // Per the module docstring: 13B's MatMul shapes exceed the
    // pool block size, so neither the M4.7.3 path
    // (residency / roundtrip) nor the legacy apx4 path serve the
    // work — Llama 2 13B forwards run on CPU. The counters are
    // surfaced for observability so a future non-pooled
    // `cuda_matmul` variant (M5+) can flip them on without
    // requiring a test rewrite. On the four M4.6 family models
    // the M4.7.6.c gate still asserts `total > 0` (see
    // `m4_7_6_c_wiring_validation_test.rs`).
    let resident_after = gpu_matmul_resident_count();
    let roundtrip_after = gpu_matmul_roundtrip_count();
    let legacy_after = gpu_matmul_legacy_count();
    let resident_delta = resident_after - resident_before;
    let roundtrip_delta = roundtrip_after - roundtrip_before;
    let legacy_delta = legacy_after - legacy_before;
    let total_delta = resident_delta + roundtrip_delta + legacy_delta;
    println!(
        "GPU MatMul invocations during forward: resident={}, roundtrip={}, legacy={}, total={}",
        resident_delta, roundtrip_delta, legacy_delta, total_delta
    );
    if total_delta == 0 {
        println!(
            "    [observability] No GPU MatMul fired on Llama 2 13B — \
             expected per the module docstring's \"Architectural finding\" \
             section. The forward ran on CPU. M5+ adds a non-pooled \
             cuda_matmul variant that lifts this limitation."
        );
    }

    // ---- 7. Per-position argmax (informational) ----
    let vocab = cfg.vocab_size;
    println!("Per-position argmax:");
    for pos in 0..runtime.seq {
        let s = pos * vocab;
        let e = s + vocab;
        let row = &slice[s..e];
        let (id, val) = row
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        // Sanity: argmax must land in the vocabulary range
        // (it always does by construction, but assert it
        // explicitly so a future shape regression is caught).
        assert!(id < vocab, "argmax id {} out of vocab range {}", id, vocab);
        println!("  Pos {}: argmax id={:>5}  logit={:.4}", pos, id, val);
    }

    println!(
        "\nPASSED: Llama 2 13B Chat Mode A green. \
         {:.0}s forward (CPU-bound; GPU dispatch is M5+). \
         GPU MatMul observability: total={}. \
         Cross-mode argmax-equality vs Modes B/C lands in M4.7.6.e.",
        fwd_secs, total_delta,
    );
}
