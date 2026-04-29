//! M4.7.4.e — TinyLlama 1.1B end-to-end smoke test with the
//! parameters spilled to the disk tier and brought back via
//! `ensure_cpu`.
//!
//! Goal: prove that with all M4.7.4.a–.d primitives wired, the
//! TinyLlama hot path still:
//!
//!   1. Spills every BF16 parameter to disk via the M4.7.4.c
//!      `migrate_all_cpu_to_disk` arm — counted in the migration
//!      report.
//!   2. Restores every spilled tensor on demand via the M4.7.4.d
//!      `ensure_cpu` Disk arm during the executor walk (the
//!      M4.7.2.c / M4.7.3.d audit already inserted `ensure_cpu`
//!      / `ensure_decoded` at every executor seam, so the
//!      restore happens automatically).
//!   3. Produces logits whose argmax matches the M4.6.1 F64
//!      ground-truth fixture on every position (4/4) — same
//!      contract as M4.7.2.d / M4.7.3.e.
//!
//! Reports total bytes written / read and the effective spill
//! and restore throughput so the user can compare the active
//! cache directory against the baseline (`ATENIA_DISK_TIER_DIR`
//! pointing at an internal NVMe is the M4.7.6 prerequisite — F:
//! USB HDD throughput is roughly 25× slower and is documented
//! as not viable for the demo).
//!
//! Marked `#[ignore]`. Run with the cache dir explicitly pinned
//! to a fast drive — the project root sits on a USB HDD on the
//! reference dev hardware:
//!
//! ```powershell
//! $env:TINYLLAMA_SAFETENSORS_PATH = "...\\model.safetensors"
//! $env:ATENIA_DISK_TIER_DIR = "D:\\Atenia\\cache_test_m4_7_4_e"
//! cargo test --test m4_7_4_e_tinyllama_disk_spill_smoke_test --release \
//!     -- --ignored --nocapture
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::llama::{
    build_llama, llama_weight_mapper, LlamaConfig, LlamaRuntime,
};
use atenia_engine::tensor::tensor::{Tensor, TensorStorage};
use atenia_engine::tensor::disk_tier;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::WeightMapper;

const EMBEDDED_TINYLLAMA_CONFIG: &str = r#"{
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
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.35.0",
  "use_cache": true,
  "vocab_size": 32000
}"#;

fn fixture_f64_reference() -> Vec<f64> {
    let path =
        PathBuf::from("tests/fixtures/tinyllama_reference/expected_logits_f64.json");
    let s = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("F64 fixture missing: {}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&s).expect("malformed F64 fixture");
    json["values"]
        .as_array()
        .expect("`values` array")
        .iter()
        .map(|v| v.as_f64().expect("number"))
        .collect()
}

#[test]
#[ignore = "requires TINYLLAMA_SAFETENSORS_PATH + ATENIA_DISK_TIER_DIR (NVMe-backed) + F64 fixture"]
fn tinyllama_with_disk_spill_argmax_matches_f64() {
    println!("\n=== TinyLlama 1.1B Disk-Spill Smoke Test (M4.7.4.e) ===\n");

    let path = env::var("TINYLLAMA_SAFETENSORS_PATH")
        .expect("Set TINYLLAMA_SAFETENSORS_PATH to TinyLlama model.safetensors");

    // Resolve the cache directory the way the disk tier itself
    // does, plus a per-test subfolder so we do not collide with
    // any concurrent process. Document the active drive in the
    // log so a regression caused by an HDD-backed cache is
    // visible at a glance.
    let cache_dir = match env::var("ATENIA_DISK_TIER_DIR") {
        Ok(v) => PathBuf::from(v),
        Err(_) => disk_tier::default_cache_dir().join("m4_7_4_e"),
    };
    fs::create_dir_all(&cache_dir).expect("create cache dir");
    println!("Cache dir for spill: {}", cache_dir.display());

    let config = LlamaConfig::from_json_str(EMBEDDED_TINYLLAMA_CONFIG).expect("parse config");
    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    // ---- 1. Build graph + load weights with BF16 storage. ----
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut graph = gb.build();
    assert_eq!(handles.param_ids.len(), 201);

    println!("Loading weights with store_params_as_bf16 = true ...");
    let load_start = Instant::now();
    let reader = SafetensorsReader::open(Path::new(&path)).expect("open safetensors");
    let mut mapper = llama_weight_mapper(&config, &handles.param_names, &handles.param_ids)
        .expect("mapper");
    let mapper_mut: &mut WeightMapper = &mut mapper;
    mapper_mut.set_store_params_as_bf16(true);
    let report = mapper.load_into(&mut graph, &reader).expect("load");
    drop(reader);
    println!(
        "Loaded {} tensors in {:.2}s",
        report.loaded,
        load_start.elapsed().as_secs_f32()
    );
    assert_eq!(report.loaded, 201);
    assert!(report.missing.is_empty());

    // ---- 2. Spill every parameter to disk. ----
    println!("Spilling parameters to disk via migrate_all_cpu_to_disk ...");
    let spill_start = Instant::now();
    let migration = graph
        .migrate_all_cpu_to_disk(&cache_dir)
        .expect("spill must succeed");
    let spill_elapsed = spill_start.elapsed();
    println!(
        "Spill: {} migrated, {} skipped, took {:.2}s",
        migration.tensors_migrated,
        migration.tensors_skipped,
        spill_elapsed.as_secs_f32()
    );

    // Pre-M4.7.4.c every CpuBf16 tensor was silently skipped.
    // After M4.7.4.c the BF16 arm is the load-bearing migration
    // path; we expect every parameter (or every CpuBf16 + Cpu
    // mix the loader produced) to land on disk. Tolerate a small
    // skip count for any non-parameter tensors the executor may
    // have populated up front (none today, but the smoke test is
    // forward-compatible).
    assert!(
        migration.tensors_migrated >= 200,
        "expected near-201 tensors spilled, got {}",
        migration.tensors_migrated
    );
    assert!(
        migration.failure.is_none(),
        "spill must not encounter per-tensor failures: {:?}",
        migration.failure
    );

    // Sum disk usage by walking the spilled handles. This also
    // proves every parameter is now Disk-backed.
    let mut total_disk_bytes: u64 = 0;
    let mut bf16_count = 0_usize;
    let mut f32_count = 0_usize;
    for &id in &handles.param_ids {
        let t = graph.nodes[id].output.as_ref().expect("param tensor");
        match &t.storage {
            TensorStorage::Disk(h) => {
                let m = fs::metadata(h.path()).expect("disk file metadata");
                total_disk_bytes += m.len();
                match h.dtype() {
                    disk_tier::DiskDtype::F32 => f32_count += 1,
                    disk_tier::DiskDtype::BF16 => bf16_count += 1,
                }
            }
            other => panic!(
                "param node {} expected Disk after spill, got {:?}",
                id, other
            ),
        }
    }
    println!(
        "Disk footprint: {:.1} MB across {} BF16 + {} F32 spilled tensors",
        total_disk_bytes as f64 / 1e6,
        bf16_count,
        f32_count
    );
    let spill_throughput_mbs =
        (total_disk_bytes as f64 / 1e6) / spill_elapsed.as_secs_f64().max(1e-6);
    println!(
        "Spill effective write throughput: {:.1} MB/s",
        spill_throughput_mbs
    );

    // ---- 3. Forward pass — restores happen lazily through
    //         ensure_cpu inside the executor. ----
    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 100.0, 200.0, 300.0]);
    println!("Running forward (parameters resident on disk; lazy restore) ...");
    let fwd_start = Instant::now();
    let outputs = graph.execute(vec![tokens]);
    let fwd_elapsed = fwd_start.elapsed();
    println!("Forward: {:.2}s", fwd_elapsed.as_secs_f32());

    let logits = &outputs[0];
    assert_eq!(logits.shape, vec![1, 4, 32_000]);
    let slice = logits.as_cpu_slice();

    // ---- 4. Logit sanity. ----
    let finite = slice.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite, slice.len(), "all logits must be finite");
    let max_abs = slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let mean_abs: f32 = slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len() as f32;
    println!("Logit stats: max |v|={:.4}  mean |v|={:.4}", max_abs, mean_abs);
    assert!(max_abs < 1000.0, "logits suspiciously large: {}", max_abs);

    // The forward effectively reads every spilled parameter at
    // least once; surface an effective restore-throughput figure
    // so a regression caused by an HDD-backed cache is visible.
    let restore_throughput_mbs =
        (total_disk_bytes as f64 / 1e6) / fwd_elapsed.as_secs_f64().max(1e-6);
    println!(
        "Forward effective restore throughput (params/forward): {:.1} MB/s",
        restore_throughput_mbs
    );

    // ---- 5. Argmax sanity vs F64 ground truth. ----
    let f64_ref = fixture_f64_reference();
    assert_eq!(f64_ref.len(), slice.len(), "F64 fixture length mismatch");

    let vocab = config.vocab_size;
    let mut argmax_match_count = 0_usize;
    for pos in 0..runtime.seq {
        let s = pos * vocab;
        let e = s + vocab;
        let a_pos = &slice[s..e];
        let f_pos = &f64_ref[s..e];

        let (a_id, _) = a_pos
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let (f_id, _) = f_pos
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let tag = if a_id == f_id {
            argmax_match_count += 1;
            "MATCH"
        } else {
            "MISMATCH"
        };
        println!(
            "  Pos {}: M4.7.4 argmax id={:>5}  F64 id={:>5}   [{}]",
            pos, a_id, f_id, tag
        );
    }
    assert_eq!(
        argmax_match_count, runtime.seq,
        "argmax mismatch under M4.7.4 disk spill on at least one position; \
         this is a real regression from the M4.7.3 / M4.6.1 baseline"
    );

    println!(
        "\nPASSED: TinyLlama M4.7.4 disk-spill forward green, argmax 4/4 vs F64. \
         Spilled {:.1} MB then restored on the fly.",
        total_disk_bytes as f64 / 1e6
    );
}
