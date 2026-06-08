//! **MOE-V3-ROUTE-1** — DeepSeek-V3-like routing **L0 mechanism** certification.
//!
//! Certifies Atenia's `src/moe/v3_router.rs` modern-routing primitives (sigmoid
//! scoring + `e_score_correction_bias` selection + group-limited top-k +
//! `routed_scaling_factor`) against a HuggingFace `DeepseekV3MoE` **float64**
//! reference on a REDUCED-DIM fixture (`fixtures/moe/v3_route_ref.*`, generated
//! by `fixtures/moe/generate_v3_route_reference.py`). The whole reduced MoE
//! block — router selection + combine weights + SwiGLU experts + ungated shared
//! expert — is reproduced from raw weights and compared end-to-end.
//!
//! This is **L0 mechanism/topology only** — NOT real V3 weights, NOT L1/L2/L3,
//! **not** the dense ADR-004 `CERTIFIED`. L4 (global F64) reserved/unreachable.
//! The router primitive is a pure reference (no runtime/loader/CUDA/Adapter
//! Toolkit); this test only *calls* it + `MoeDenseExpert`.
//!
//! Reproduce the fixture:
//!   python fixtures/moe/generate_v3_route_reference.py

use atenia_engine::moe::dense::MoeDenseExpert;
use atenia_engine::moe::v3_router::{v3_route, ScoringFunc, V3RouterConfig};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::path::PathBuf;

const GATE: f64 = 1e-3; // L0 mechanism drift bound (HF f32 router vs Atenia f64 + f64 experts)

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn meta() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("v3_route_ref.json")).unwrap())
        .expect("v3_route_ref.json (run the generator first)")
}

fn u(j: &serde_json::Value, k: &str) -> usize {
    j[k].as_u64().unwrap() as usize
}
fn f64v(j: &serde_json::Value, k: &str) -> Vec<f64> {
    j[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// `y = W·x`, `W` row-major `[rows, cols]`, f64 accumulation.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let mut acc = 0.0_f64;
            for c in 0..cols {
                acc += w[r * cols + c] as f64 * x[c] as f64;
            }
            acc as f32
        })
        .collect()
}

#[test]
fn deepseek_v3_route_scale_certifies() {
    unsafe {
        std::env::set_var("ATENIA_ENABLE_MOE", "1");
    }
    let j = meta();
    let hidden = u(&j, "hidden_size");
    let inter = u(&j, "moe_intermediate_size");
    let shared_inter = u(&j, "shared_intermediate_size");
    let n = u(&j, "n_routed_experts");
    let tokens = u(&j, "tokens");

    let cfg = V3RouterConfig {
        n_routed_experts: n,
        top_k: u(&j, "top_k"),
        n_group: u(&j, "n_group"),
        topk_group: u(&j, "topk_group"),
        routed_scaling_factor: j["routed_scaling_factor"].as_f64().unwrap(),
        norm_topk_prob: j["norm_topk_prob"].as_bool().unwrap(),
        scoring_func: ScoringFunc::parse(j["scoring_func"].as_str().unwrap()).unwrap(),
    };

    let r = SafetensorsReader::open(&fixture_dir().join("v3_route_ref.safetensors")).unwrap();
    let get = |name: &str| r.get(name).unwrap().to_vec_f32().unwrap();

    let router_w = get("router.weight"); // [n, hidden]
    let bias = get("router.bias"); // [n]
    let experts: Vec<MoeDenseExpert> = (0..n)
        .map(|e| {
            MoeDenseExpert::new(
                hidden,
                inter,
                get(&format!("expert.{e}.w_gate")),
                get(&format!("expert.{e}.w_up")),
                get(&format!("expert.{e}.w_down")),
            )
            .unwrap()
        })
        .collect();
    let shared = MoeDenseExpert::new(
        hidden,
        shared_inter,
        get("shared.w_gate"),
        get("shared.w_up"),
        get("shared.w_down"),
    )
    .unwrap();
    let hidden_states = get("hidden"); // [tokens, hidden]

    let ref_dense = f64v(&j, "dense_combine_weights"); // [tokens * n]
    let ref_block = f64v(&j, "block_out"); // [tokens * hidden]
    let ref_selected: Vec<Vec<usize>> = j["selected_experts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect())
        .collect();

    let mut worst_w = 0.0_f64;
    let mut worst_block = 0.0_f64;
    let mut set_mismatch: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
    let mut min_margin = f64::INFINITY;

    for t in 0..tokens {
        let h = &hidden_states[t * hidden..(t + 1) * hidden];
        let logits = matvec(&router_w, n, hidden, h);
        let routing = v3_route(&logits, &bias, &cfg).unwrap();
        min_margin = min_margin.min(routing.selection_margin);

        // (1) router SET EQUALITY vs HF.
        if routing.indices != ref_selected[t] {
            set_mismatch.push((t, routing.indices.clone(), ref_selected[t].clone()));
        }

        // (2) dense combine-weight diff.
        for e in 0..n {
            let d = (routing.dense_weights[e] as f64 - ref_dense[t * n + e]).abs();
            worst_w = worst_w.max(d);
        }

        // (3) full MoE-block forward: Σ w·expert(h) + shared(h), vs HF block out.
        let mut out = vec![0.0_f64; hidden];
        for (slot, &e) in routing.indices.iter().enumerate() {
            let y = experts[e].forward(h).unwrap();
            let w = routing.weights[slot] as f64;
            for d in 0..hidden {
                out[d] += w * y[d] as f64;
            }
        }
        let s = shared.forward(h).unwrap();
        for d in 0..hidden {
            out[d] += s[d] as f64;
            let diff = (out[d] - ref_block[t * hidden + d]).abs();
            worst_block = worst_block.max(diff);
        }
    }

    eprintln!(
        "MOE-V3-ROUTE-1 L0: DeepSeek-V3-like routing vs HF f64 | tokens={tokens} n_routed={n} \
         groups={}/{} top_k={} | worst combine-weight diff={:.3e} | worst block max_abs_diff={:.3e} \
         | router set-equality mismatches={:?} | min selection margin={:.4e}",
        cfg.topk_group, cfg.n_group, cfg.top_k, worst_w, worst_block, set_mismatch, min_margin
    );

    // Determinism: the router is bit-identical on a second pass.
    for t in 0..tokens {
        let h = &hidden_states[t * hidden..(t + 1) * hidden];
        let logits = matvec(&router_w, n, hidden, h);
        let a = v3_route(&logits, &bias, &cfg).unwrap();
        let b = v3_route(&logits, &bias, &cfg).unwrap();
        assert_eq!(a, b, "v3_route must be deterministic (token {t})");
    }

    // Hard gates — fail loud (NOT certifying the mechanism if any fails).
    assert!(
        set_mismatch.is_empty(),
        "V3 router selected-expert SET differs from the HF f64 reference: {set_mismatch:?}"
    );
    assert!(worst_w < GATE, "combine-weight diff {worst_w:.3e} >= {GATE:.0e}");
    assert!(worst_block < GATE, "block max_abs_diff {worst_block:.3e} >= {GATE:.0e}");

    eprintln!(
        "MOE-V3-ROUTE-1 RESULT: DeepSeek-V3-like routing mechanism L0 PASS (vs HF f64, reduced-dim; \
         router set-equality {tokens}/{tokens}, worst block {worst_block:.3e} < {GATE:.0e}, \
         deterministic). L0 mechanism/topology only — not real-weight certified, not L1/L2/L3, \
         not dense ADR-004 CERTIFIED; L4 reserved/unreachable."
    );
}
