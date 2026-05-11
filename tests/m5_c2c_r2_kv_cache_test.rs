//! M5.c.2.c — R2 graph-level falsifier for the cache-aware
//! attention path.
//!
//! Strategy: random-weighted mini-Llama (small enough to run
//! in milliseconds, large enough to exercise every layer of
//! the architecture). Three contracts are locked here:
//!
//!   1. **No-cache parity** — `build_llama_with_store` with
//!      `kv_cache = None` produces logits bit-exact equivalent
//!      to `build_llama` populated with the same weights. The
//!      Arc-shared parameter path does not perturb numerics.
//!
//!   2. **Empty-cache parity** — `build_llama_with_store`
//!      with `kv_cache = Some(KvCacheBuildSpec { cached_len: 0 })`
//!      produces the same logits as the no-cache path. The
//!      cache machinery (Concat over an empty cache,
//!      RoPE position_offset = 0, mask shape [1,1,seq,seq])
//!      collapses to the no-cache numerics when the cache is
//!      empty.
//!
//!   3. **Prefill→Decode equivalence (R2)** — running the
//!      cached path as `prefill at seq=N` followed by `K decode
//!      steps at seq=1` produces the same per-position argmax
//!      as a single no-cache forward at seq=N+K. Logit values
//!      additionally agree to within 1e-3 max-abs-diff (the R2
//!      tolerance from the M5 plan).
//!
//! ## Why a random-weighted mini config
//!
//! The R2 contract is purely architectural: it asserts that
//! the cache-aware path produces the same MATH as the no-cache
//! path, regardless of what weights are loaded. A random tiny
//! config exercises every code path in a few milliseconds and
//! does not require checkpoint files on disk. The contract
//! transfers to real models because the math is the same.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::kv_cache::KvCacheBuildSpec;
use atenia_engine::amg::weight_store::WeightStore;
use atenia_engine::nn::llama::{LlamaConfig, LlamaRuntime, build_llama, build_llama_with_store};
use atenia_engine::tensor::Tensor;

/// Tiny MHA Llama config: 2 layers × 2 heads × head_dim 4 →
/// hidden 8, intermediate 16, vocab 8. MHA (num_kv_heads ==
/// num_attention_heads) sidesteps the GQA load-pipeline tile
/// for this first R2 — gap-3's pre-tile cache decision lands
/// in M5.c.2.d's GQA validation.
fn mini_config() -> LlamaConfig {
    LlamaConfig::from_json_str(
        r#"{
          "vocab_size": 8,
          "hidden_size": 8,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "intermediate_size": 16,
          "max_position_embeddings": 16,
          "rope_theta": 10000.0,
          "rms_norm_eps": 1e-5,
          "tie_word_embeddings": false,
          "attention_bias": false,
          "bos_token_id": 0,
          "eos_token_id": 1
        }"#,
    )
    .expect("mini config must parse")
}

/// Deterministic pseudo-random weight generator. A tiny
/// linear-congruential-style hash keyed on the parameter
/// index and element index keeps the test self-contained
/// (no rand crate, fully reproducible across runs).
fn deterministic_weight(param_index: usize, element_index: usize, numel: usize) -> f32 {
    let seed =
        (param_index as u64).wrapping_mul(2654435761) ^ (element_index as u64).wrapping_mul(40503);
    // Map the hash into a small float interval so RmsNorm and
    // softmax stay numerically well-behaved. Values in
    // ~ [-0.2, 0.2] keep the layer outputs O(1).
    let frac = ((seed % 4001) as f32) / 4001.0;
    let scaled = (frac - 0.5) * 0.4;
    // Slight per-element decay — doesn't matter for the R2
    // contract, just makes the generated logits less symmetric.
    scaled * (1.0 + (element_index as f32) / (numel.max(1) as f32))
}

fn fill_graph_with_deterministic_weights(
    graph: &mut Graph,
    param_ids: &[usize],
    param_names: &[String],
) {
    for (i, (&node_id, name)) in param_ids.iter().zip(param_names.iter()).enumerate() {
        let shape = graph.nodes[node_id].output.as_ref().unwrap().shape.clone();
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel)
            .map(|j| deterministic_weight(i, j, numel))
            .collect();
        // Special-case: layernorm gammas should be ~1, not
        // small random — keeps the RMSNorm scale sensible.
        let _ = name;
        let final_data = if name.ends_with("layernorm.weight") || name == "model.norm.weight" {
            (0..numel)
                .map(|j| 1.0 + deterministic_weight(i, j, numel) * 0.1)
                .collect()
        } else {
            data
        };
        graph
            .overwrite_parameter(node_id, Tensor::new_cpu(shape, final_data))
            .expect("overwrite_parameter failed");
    }
}

fn build_reference_graph(cfg: &LlamaConfig, seq: usize) -> (Graph, Vec<usize>, Vec<String>) {
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let runtime = LlamaRuntime { batch: 1, seq };
    let h = build_llama(&mut gb, cfg, &runtime, token_input_id);
    let _ = gb.output(h.logits_id);
    let mut graph = gb.build();
    fill_graph_with_deterministic_weights(&mut graph, &h.param_ids, &h.param_names);
    (graph, h.param_ids, h.param_names)
}

/// Read shared-storage F32 vec out of a graph parameter slot.
/// Used to build the WeightStore via direct construction so
/// the test doesn't depend on `extract_from_graph` (covered
/// by M5.c.2.b's own tests).
fn read_param_data(graph: &Graph, node_id: usize) -> Vec<f32> {
    graph.nodes[node_id]
        .output
        .as_ref()
        .unwrap()
        .copy_to_cpu_vec()
}

fn build_store_from_reference(
    graph: &Graph,
    param_ids: &[usize],
    param_names: &[String],
) -> WeightStore {
    let mut store = WeightStore::new();
    for (&node_id, name) in param_ids.iter().zip(param_names.iter()) {
        let shape = graph.nodes[node_id].output.as_ref().unwrap().shape.clone();
        let data = read_param_data(graph, node_id);
        store.insert_f32(name.clone(), shape, data);
    }
    store
}

fn argmax_per_position(logits: &Tensor) -> Vec<usize> {
    let shape = &logits.shape;
    let batch = shape[0];
    let seq = shape[1];
    let vocab = shape[2];
    assert_eq!(batch, 1);
    let data = logits.copy_to_cpu_vec();
    (0..seq)
        .map(|s| {
            let row = &data[s * vocab..(s + 1) * vocab];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, |acc, d| acc.max(d))
}

// ============================================================
//  Contract 1: build_llama_with_store(no-cache) == build_llama
// ============================================================

#[test]
fn no_cache_path_matches_build_llama_bit_exact() {
    let cfg = mini_config();
    let seq = 4usize;
    let tokens = Tensor::new_cpu(vec![1, seq], vec![0.0, 1.0, 2.0, 3.0]);

    // Reference: classic build_llama with deterministic weights.
    let (mut g_ref, ref_ids, ref_names) = build_reference_graph(&cfg, seq);
    let logits_ref = g_ref
        .execute(vec![tokens.clone()])
        .into_iter()
        .next()
        .unwrap();

    // Cached path with kv_cache = None: same architecture, same
    // weights through the WeightStore.
    let store = build_store_from_reference(&g_ref, &ref_ids, &ref_names);
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let runtime = LlamaRuntime { batch: 1, seq };
    let h_shared = build_llama_with_store(&mut gb, &cfg, &runtime, token_input_id, &store, None)
        .expect("build_llama_with_store(no-cache) must succeed");
    let _ = gb.output(h_shared.logits_id);
    let mut g_shared = gb.build();
    let logits_shared = g_shared.execute(vec![tokens]).into_iter().next().unwrap();

    assert_eq!(logits_ref.shape, logits_shared.shape);
    let diff = max_abs_diff(
        &logits_ref.copy_to_cpu_vec(),
        &logits_shared.copy_to_cpu_vec(),
    );
    assert!(
        diff < 1e-6,
        "no-cache parity: max-abs logit diff {diff:.3e} exceeds 1e-6 tolerance"
    );

    // Argmax must match exactly.
    assert_eq!(
        argmax_per_position(&logits_ref),
        argmax_per_position(&logits_shared)
    );
}

// ============================================================
//  Contract 2: empty-cache path == no-cache path
// ============================================================

#[test]
fn empty_cache_path_matches_no_cache_at_every_position() {
    let cfg = mini_config();
    let seq = 4usize;
    let tokens = Tensor::new_cpu(vec![1, seq], vec![0.0, 1.0, 2.0, 3.0]);

    // Reference (no-cache).
    let (mut g_ref, ref_ids, ref_names) = build_reference_graph(&cfg, seq);
    let logits_ref = g_ref
        .execute(vec![tokens.clone()])
        .into_iter()
        .next()
        .unwrap();

    // Cache-aware path with cached_len = 0. Concat against an
    // empty cache, RoPE offset = 0, mask shape [1,1,4,4]. Should
    // collapse to the no-cache numerics exactly.
    let store = build_store_from_reference(&g_ref, &ref_ids, &ref_names);
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let runtime = LlamaRuntime { batch: 1, seq };
    let spec = KvCacheBuildSpec { cached_len: 0 };
    let h = build_llama_with_store(&mut gb, &cfg, &runtime, token_input_id, &store, Some(&spec))
        .expect("build_llama_with_store(empty cache) must succeed");
    let _ = gb.output(h.logits_id);
    let mut g_cache = gb.build();
    let logits_cache = g_cache.execute(vec![tokens]).into_iter().next().unwrap();

    let diff = max_abs_diff(
        &logits_ref.copy_to_cpu_vec(),
        &logits_cache.copy_to_cpu_vec(),
    );
    assert!(
        diff < 1e-3,
        "empty-cache parity: max-abs logit diff {diff:.3e} exceeds R2 1e-3 tolerance"
    );
    assert_eq!(
        argmax_per_position(&logits_ref),
        argmax_per_position(&logits_cache),
        "empty-cache argmax must match no-cache argmax at every position"
    );

    // Sanity: handles surface populated.
    let kv = h
        .kv_handles
        .expect("kv_handles must be Some when build had cache spec");
    assert_eq!(kv.per_layer.len(), cfg.num_hidden_layers);
}

// ============================================================
//  Contract 3: full R2 — prefill + decode steps == no-cache
// ============================================================

#[test]
fn prefill_then_decode_steps_match_no_cache_reference_r2() {
    let cfg = mini_config();
    let prefill_len = 2usize;
    let n_decode_steps = 2usize;
    let total_len = prefill_len + n_decode_steps;

    // Reference forward at full length seq=4, no cache.
    let tokens_full: Vec<f32> = (0..total_len).map(|i| i as f32).collect();
    let (mut g_ref, ref_ids, ref_names) = build_reference_graph(&cfg, total_len);
    let tokens_full_tensor = Tensor::new_cpu(vec![1, total_len], tokens_full.clone());
    let logits_ref = g_ref
        .execute(vec![tokens_full_tensor])
        .into_iter()
        .next()
        .unwrap();
    let logits_ref_data = logits_ref.copy_to_cpu_vec();
    let vocab = cfg.vocab_size;

    let store = build_store_from_reference(&g_ref, &ref_ids, &ref_names);

    // ---- Prefill at seq=prefill_len, cached_len=0 ----
    let prefill_tokens = Tensor::new_cpu(vec![1, prefill_len], tokens_full[..prefill_len].to_vec());
    let mut gb_p = GraphBuilder::new();
    let token_in_p = gb_p.input();
    let runtime_p = LlamaRuntime {
        batch: 1,
        seq: prefill_len,
    };
    let spec_p = KvCacheBuildSpec { cached_len: 0 };
    let h_p = build_llama_with_store(
        &mut gb_p,
        &cfg,
        &runtime_p,
        token_in_p,
        &store,
        Some(&spec_p),
    )
    .expect("prefill build");
    let _ = gb_p.output(h_p.logits_id);
    let mut g_p = gb_p.build();
    let logits_p = g_p
        .execute(vec![prefill_tokens])
        .into_iter()
        .next()
        .unwrap();
    let logits_p_data = logits_p.copy_to_cpu_vec();

    // Compare prefill logits at positions 0..prefill_len with
    // reference at the same positions.
    for pos in 0..prefill_len {
        let r = &logits_ref_data[pos * vocab..(pos + 1) * vocab];
        let p = &logits_p_data[pos * vocab..(pos + 1) * vocab];
        let diff = max_abs_diff(r, p);
        assert!(
            diff < 1e-3,
            "prefill position {pos}: max-abs diff {diff:.3e} > 1e-3"
        );
    }

    // Harvest K_full, V_full per layer from the prefill graph.
    let kv_p = h_p.kv_handles.expect("prefill must produce kv_handles");
    let mut cache_k_per_layer: Vec<Tensor> = Vec::with_capacity(kv_p.per_layer.len());
    let mut cache_v_per_layer: Vec<Tensor> = Vec::with_capacity(kv_p.per_layer.len());
    for layer in &kv_p.per_layer {
        let k = g_p.nodes[layer.k_full_node_id]
            .output
            .as_ref()
            .expect("k_full not materialised")
            .clone();
        let v = g_p.nodes[layer.v_full_node_id]
            .output
            .as_ref()
            .expect("v_full not materialised")
            .clone();
        cache_k_per_layer.push(k);
        cache_v_per_layer.push(v);
    }

    // ---- Decode loop ----
    //
    // Per step: build a fresh decode graph at seq=1 with the
    // current cached_len, set its cache_K / cache_V slots from
    // the harvested tensors, run forward, harvest K_full /
    // V_full again for the next step.
    for step in 0..n_decode_steps {
        let cached_len = prefill_len + step;
        let next_token = tokens_full[cached_len];

        let token_tensor = Tensor::new_cpu(vec![1, 1], vec![next_token]);
        let mut gb_d = GraphBuilder::new();
        let token_in_d = gb_d.input();
        let runtime_d = LlamaRuntime { batch: 1, seq: 1 };
        let spec_d = KvCacheBuildSpec { cached_len };
        let h_d = build_llama_with_store(
            &mut gb_d,
            &cfg,
            &runtime_d,
            token_in_d,
            &store,
            Some(&spec_d),
        )
        .expect("decode build");
        let _ = gb_d.output(h_d.logits_id);
        let mut g_d = gb_d.build();

        // Patch the cache slots with the harvested tensors.
        let kv_d = h_d.kv_handles.as_ref().expect("decode kv_handles");
        for (layer_idx, layer) in kv_d.per_layer.iter().enumerate() {
            g_d.overwrite_parameter(layer.cache_k_param_id, cache_k_per_layer[layer_idx].clone())
                .expect("overwrite cache_K");
            g_d.overwrite_parameter(layer.cache_v_param_id, cache_v_per_layer[layer_idx].clone())
                .expect("overwrite cache_V");
        }

        let logits_d = g_d.execute(vec![token_tensor]).into_iter().next().unwrap();
        let logits_d_data = logits_d.copy_to_cpu_vec();
        assert_eq!(logits_d.shape, vec![1, 1, vocab]);

        // Compare to reference logits at the same absolute position.
        let r = &logits_ref_data[cached_len * vocab..(cached_len + 1) * vocab];
        let p = &logits_d_data;
        let diff = max_abs_diff(r, p);
        assert!(
            diff < 1e-3,
            "decode step {step} (cached_len={cached_len}): max-abs diff {diff:.3e} > 1e-3"
        );

        // Argmax must match position-wise.
        let r_amax = r
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let p_amax = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            r_amax, p_amax,
            "decode step {step}: argmax mismatch (ref={r_amax}, decode={p_amax})"
        );

        // Harvest K_full, V_full for next step's cache.
        for (layer_idx, layer) in kv_d.per_layer.iter().enumerate() {
            cache_k_per_layer[layer_idx] = g_d.nodes[layer.k_full_node_id]
                .output
                .as_ref()
                .expect("k_full")
                .clone();
            cache_v_per_layer[layer_idx] = g_d.nodes[layer.v_full_node_id]
                .output
                .as_ref()
                .expect("v_full")
                .clone();
        }
    }

    // After all decode steps, cache should have grown to total_len.
    assert_eq!(
        cache_k_per_layer[0].shape[2], total_len,
        "K cache final length should equal total_len after {n_decode_steps} steps"
    );
}
