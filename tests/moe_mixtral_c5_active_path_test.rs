//! **MIXTRAL-CERT-3 / C5** — Mixtral-8x7B-v0.1 active-path parity (ADR-007), reaching L3.
//!
//! ADR-007 C5 certifies the **real model, as actually run**, end-to-end on a
//! canonical input vs an external reference over the active subgraph. A global F64
//! forward is infeasible (~374 GB), so the reference is computed **one decoder layer
//! at a time in float64** using HuggingFace's own Mixtral layer module
//! (`fixtures/moe/generate_mixtral_c5_reference.py`) — the F64 form of C5, never the
//! whole model in F64 (NOT L4). The reference driver is validated against the tiny
//! `mixtral_scale` fixture (2.021e-8 vs HF f64) before use.
//!
//! This harness runs Atenia's controlled `MoeRuntime` full forward on the real
//! Mixtral-8x7B-v0.1 via the **disk expert-tier** (`ATENIA_MOE_EXPERT_TIER=disk`,
//! optionally persistent — experts stream from NVMe, RAM ~ backend + active set) and
//! gates it against that f64 reference: end-to-end `max_abs_diff < 0.5` (ADR-004 bar,
//! unchanged) + exact per-position argmax + determinism. **Test-only** — no runtime /
//! loader / MoeRuntime / Adapter Toolkit / numerics change; it only *calls* the runtime.
//!
//! `#[ignore]` + env `MIXTRAL_DIR`. Reproduce the reference + run:
//!   python fixtures/moe/generate_mixtral_c5_reference.py real <dir> fixtures/moe
//!   $env:ATENIA_MOE_EXPERT_TIER="disk"; $env:ATENIA_MOE_TIER_PERSIST="1"
//!   $env:ATENIA_DISK_TIER_DIR="D:\mixtral_tier"; $env:MIXTRAL_DIR="<dir>"
//!   cargo test --test moe_mixtral_c5_active_path_test --release -- \
//!     --ignored mixtral_real_c5_active_path --nocapture

use std::path::PathBuf;
use std::time::Instant;

use atenia_engine::moe::MoeRuntime;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

const ADR_004_GATE: f32 = 0.5;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

#[test]
#[ignore = "needs the real ~87 GB Mixtral-8x7B-v0.1 checkpoint via MIXTRAL_DIR + an NVMe tier dir"]
fn mixtral_real_c5_active_path() {
    let model_dir = std::env::var("MIXTRAL_DIR").expect("set MIXTRAL_DIR to the Mixtral checkout");

    // Controlled MoE opt-in + disk-tier residency (so the 46.7B model fits RAM).
    // The persistent tier / dir is set by the runner via env (ATENIA_MOE_TIER_PERSIST,
    // ATENIA_DISK_TIER_DIR). SAFETY: single-threaded test owning these env toggles.
    unsafe {
        std::env::set_var("ATENIA_ENABLE_MOE", "1");
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
        if std::env::var("ATENIA_MOE_EXPERT_TIER").is_err() {
            std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
        }
    }

    // --- committed C5 f64 reference (one layer at a time) ---
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("mixtral_c5_ref.json")).unwrap(),
    )
    .expect("C5 reference json (run the generator first)");
    let input_ids: Vec<u32> = meta["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let vocab = meta["vocab_size"].as_u64().unwrap() as usize;
    let seq = meta["seq"].as_u64().unwrap() as usize;
    let ref_argmax: Vec<usize> = meta["argmax_per_position"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let refr = SafetensorsReader::open(&fixture_dir().join("mixtral_c5_ref.safetensors")).unwrap();
    let ref_logits = refr.get("logits").unwrap().to_vec_f32().unwrap();
    assert_eq!(ref_logits.len(), seq * vocab, "reference logits dims");

    // --- Atenia: real full forward through the controlled MoE runtime (disk tier) ---
    let t_load = Instant::now();
    let rt = MoeRuntime::load_from_dir(&PathBuf::from(&model_dir))
        .unwrap_or_else(|e| panic!("MoeRuntime must load the real Mixtral-8x7B-v0.1: {e}"));
    let load_s = t_load.elapsed().as_secs_f64();

    let t_fwd = Instant::now();
    let got = rt.forward_logits(&input_ids);
    let fwd_s = t_fwd.elapsed().as_secs_f64();
    assert_eq!(got.len(), seq * vocab, "atenia logits dims");

    // === C5 — active-path parity (gate < 0.5 + exact per-position argmax) ===
    let mut worst = 0.0_f32;
    let mut worst_pos = 0usize;
    let mut argmax_mismatch: Vec<(usize, usize, usize)> = Vec::new();
    for pos in 0..seq {
        let a = &got[pos * vocab..(pos + 1) * vocab];
        let r = &ref_logits[pos * vocab..(pos + 1) * vocab];
        let d = a.iter().zip(r).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        if d > worst {
            worst = d;
            worst_pos = pos;
        }
        let (ga, ra) = (argmax(a), ref_argmax[pos]);
        if ga != ra {
            argmax_mismatch.push((pos, ga, ra));
        }
    }
    println!(
        "MIXTRAL-C5 real Mixtral-8x7B-v0.1: seq={seq} input_ids={input_ids:?} | load={load_s:.1}s \
         forward={fwd_s:.1}s | worst max_abs_diff={worst:.3e} (pos {worst_pos}) | \
         ref argmax={ref_argmax:?} | argmax_mismatches={argmax_mismatch:?}"
    );

    // Determinism: a second forward is bit-identical.
    let got2 = rt.forward_logits(&input_ids);
    assert_eq!(got, got2, "MoeRuntime forward must be deterministic");

    // Hard gates — if either fails, do NOT certify L3 (asserts fail loudly).
    assert!(
        argmax_mismatch.is_empty(),
        "C5 FAILED — per-position argmax differs from the f64 reference (not certifying L3): {argmax_mismatch:?}"
    );
    assert!(
        worst < ADR_004_GATE,
        "C5 FAILED — end-to-end max_abs_diff {worst:.3e} >= {ADR_004_GATE} (not certifying L3)"
    );

    println!(
        "MIXTRAL-C5 RESULT: Mixtral-8x7B-v0.1 C5 PASS (active-path, real weights, disk tier, \
         worst {worst:.3e} < {ADR_004_GATE}; argmax exact {seq}/{seq}; deterministic). \
         F64 reference, one layer at a time (NOT L4). -> ADR-007 L3 for Mixtral-8x7B-v0.1."
    );
}
