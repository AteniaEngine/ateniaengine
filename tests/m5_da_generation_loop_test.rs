//! M5.d.a — generation-loop synthetic correctness tests.
//!
//! Random-weighted mini-Llama (same architectural config as
//! the M5.c.2.c R2 falsifier). The contracts are:
//!
//!   1. **Argmax-consistency with R2** — running the
//!      generation loop for K steps produces the same per-
//!      step token IDs as taking argmax over a no-cache
//!      forward at each prefix length. This extends the
//!      M5.c.2.c R2 contract from "logits match" to "the
//!      decision the loop makes matches".
//!
//!   2. **EOS halt** — when greedy argmax happens to land
//!      on the EOS id (we engineer a config where this is
//!      easy to hit by setting eos to the dominant logit),
//!      generation halts immediately and the EOS token is
//!      reported as `is_eos = true` in the stream.
//!
//!   3. **max_new_tokens cap** — when EOS never fires, the
//!      loop respects the budget and returns exactly the
//!      requested count.
//!
//!   4. **Streaming sink ordering** — `CollectingTokenSink`
//!      receives tokens in step order, with each token's
//!      `step` field matching the array index.
//!
//! Real-checkpoint generation, GQA pre-tile validation,
//! determinism fixture, and 13B Arc-sharing proof land in
//! M5.d.b / M5.d.c.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::weight_store::WeightStore;
use atenia_engine::nn::llama::{
    CollectingTokenSink, GenerationConfig, LlamaConfig, LlamaRuntime, build_llama, generate_greedy,
};
use atenia_engine::tensor::Tensor;

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

fn deterministic_weight(param_index: usize, element_index: usize, numel: usize) -> f32 {
    let seed =
        (param_index as u64).wrapping_mul(2654435761) ^ (element_index as u64).wrapping_mul(40503);
    let frac = ((seed % 4001) as f32) / 4001.0;
    let scaled = (frac - 0.5) * 0.4;
    scaled * (1.0 + (element_index as f32) / (numel.max(1) as f32))
}

fn build_reference(cfg: &LlamaConfig, seq: usize) -> (Graph, Vec<usize>, Vec<String>) {
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let runtime = LlamaRuntime { batch: 1, seq };
    let h = build_llama(&mut gb, cfg, &runtime, token_input_id);
    let _ = gb.output(h.logits_id);
    let mut graph = gb.build();
    for (i, (&node_id, name)) in h.param_ids.iter().zip(h.param_names.iter()).enumerate() {
        let shape = graph.nodes[node_id].output.as_ref().unwrap().shape.clone();
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = if name.ends_with("layernorm.weight") || name == "model.norm.weight" {
            (0..numel)
                .map(|j| 1.0 + deterministic_weight(i, j, numel) * 0.1)
                .collect()
        } else {
            (0..numel)
                .map(|j| deterministic_weight(i, j, numel))
                .collect()
        };
        graph
            .overwrite_parameter(node_id, Tensor::new_cpu(shape, data))
            .unwrap();
    }
    (graph, h.param_ids, h.param_names)
}

fn store_from_graph(g: &Graph, ids: &[usize], names: &[String]) -> WeightStore {
    let mut store = WeightStore::new();
    for (&node_id, name) in ids.iter().zip(names.iter()) {
        let t = g.nodes[node_id].output.as_ref().unwrap();
        store.insert_f32(name.clone(), t.shape.clone(), t.copy_to_cpu_vec());
    }
    store
}

#[test]
fn greedy_loop_matches_per_step_no_cache_argmax() {
    // Contract 1: the loop's tokens equal what you'd get
    // by running a no-cache forward at each growing prefix
    // and taking argmax of the last-position logits.
    let cfg = mini_config();
    let prompt: Vec<u32> = vec![3, 2, 4]; // arbitrary ids in [0..vocab=8)
    let n_new = 4usize;

    // Build store from a reference graph at any seq (we only
    // need the weights; weights come out of the reference's
    // overwrite_parameter step).
    let (g_ref_seed, ids, names) = build_reference(&cfg, prompt.len());
    let store = store_from_graph(&g_ref_seed, &ids, &names);

    // Run generation.
    let gen_cfg = GenerationConfig {
        max_new_tokens: n_new,
        eos_token_ids: vec![9999], // outside vocab → never fires
    };
    let mut sink = CollectingTokenSink::default();
    // Trivial decode_token (returns id as decimal string).
    let decode = |id: u32| format!("[{id}]");
    let generated = generate_greedy(&cfg, &store, &prompt, &gen_cfg, decode, &mut sink)
        .expect("generate must succeed");

    assert_eq!(generated.len(), n_new, "loop must return n_new tokens");
    assert_eq!(sink.tokens.len(), n_new, "sink must receive n_new tokens");

    // Reference expectation: at each step k, the predicted
    // next token is argmax of the no-cache forward logits at
    // the LAST position when the input is `prompt + generated[..k]`.
    let mut full_input: Vec<u32> = prompt.clone();
    for (k, &gen_id) in generated.iter().enumerate() {
        // Build a fresh no-cache graph at seq=full_input.len()
        // and read the argmax of the last position.
        let seq = full_input.len();
        let (mut g_ref, ids, names) = build_reference(&cfg, seq);
        // Patch the same weights (deterministic, but
        // build_reference already populated them). Just confirm
        // the param shapes match the seed run.
        let _ = ids;
        let _ = names;
        let token_input =
            Tensor::new_cpu(vec![1, seq], full_input.iter().map(|&t| t as f32).collect());
        let logits = g_ref.execute(vec![token_input]).into_iter().next().unwrap();
        let logits_data = logits.copy_to_cpu_vec();
        let vocab = cfg.vocab_size;
        let last_row = &logits_data[(seq - 1) * vocab..seq * vocab];
        let expected_id = last_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;

        assert_eq!(
            gen_id, expected_id,
            "step {k}: loop argmax {gen_id} differs from no-cache reference argmax {expected_id} \
             (full_input = {full_input:?})"
        );

        // Also check the sink event matches.
        assert_eq!(sink.tokens[k].step, k);
        assert_eq!(sink.tokens[k].token_id, gen_id);
        assert!(!sink.tokens[k].is_eos, "shouldn't see EOS in this run");
        assert_eq!(sink.tokens[k].text, format!("[{gen_id}]"));

        full_input.push(gen_id);
    }
}

#[test]
fn eos_halts_generation_immediately() {
    // Contract 2: when greedy argmax lands on the configured
    // eos_token_id, the loop returns at that step and the
    // event has is_eos = true. We pick eos = whatever the
    // first generated token is, so the very first decode
    // step terminates.
    let cfg = mini_config();
    let prompt: Vec<u32> = vec![5, 6];

    let (g_seed, ids, names) = build_reference(&cfg, prompt.len());
    let store = store_from_graph(&g_seed, &ids, &names);

    // Probe the first generated token by running the
    // reference forward and taking the last-row argmax.
    let (mut g_ref, _, _) = build_reference(&cfg, prompt.len());
    let token_input = Tensor::new_cpu(
        vec![1, prompt.len()],
        prompt.iter().map(|&t| t as f32).collect(),
    );
    let logits = g_ref.execute(vec![token_input]).into_iter().next().unwrap();
    let logits_data = logits.copy_to_cpu_vec();
    let vocab = cfg.vocab_size;
    let last_row = &logits_data[(prompt.len() - 1) * vocab..prompt.len() * vocab];
    let first_id = last_row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;

    let gen_cfg = GenerationConfig {
        max_new_tokens: 100,
        eos_token_ids: vec![first_id],
    };
    let mut sink = CollectingTokenSink::default();
    let generated = generate_greedy(
        &cfg,
        &store,
        &prompt,
        &gen_cfg,
        |id| format!("[{id}]"),
        &mut sink,
    )
    .unwrap();

    assert_eq!(
        generated,
        vec![first_id],
        "EOS must be emitted exactly once"
    );
    assert_eq!(sink.tokens.len(), 1);
    assert!(
        sink.tokens[0].is_eos,
        "first stream event must be flagged is_eos"
    );
    // EOS token should not have a rendered text fragment.
    assert!(sink.tokens[0].text.is_empty());
}

#[test]
fn max_new_tokens_caps_runaway_generation() {
    // Contract 3: when EOS never fires (eos id outside the
    // vocabulary), generation hits exactly `max_new_tokens`
    // and stops.
    let cfg = mini_config();
    let prompt: Vec<u32> = vec![7];
    let cap = 5usize;

    let (g_seed, ids, names) = build_reference(&cfg, prompt.len());
    let store = store_from_graph(&g_seed, &ids, &names);

    let gen_cfg = GenerationConfig {
        max_new_tokens: cap,
        eos_token_ids: vec![9999],
    };
    let mut sink = CollectingTokenSink::default();
    let generated = generate_greedy(
        &cfg,
        &store,
        &prompt,
        &gen_cfg,
        |id| format!("[{id}]"),
        &mut sink,
    )
    .unwrap();

    assert_eq!(generated.len(), cap, "must respect max_new_tokens cap");
    assert_eq!(sink.tokens.len(), cap);
    assert!(sink.tokens.iter().all(|t| !t.is_eos));
    // step indices must be 0..cap in order.
    for (i, t) in sink.tokens.iter().enumerate() {
        assert_eq!(t.step, i, "step index must equal array index");
    }
}

#[test]
fn streaming_sink_receives_each_token_in_order() {
    // Contract 4: closure-based sink (FnMut) sees the same
    // tokens as the returned Vec, in order.
    let cfg = mini_config();
    let prompt: Vec<u32> = vec![2, 3];

    let (g_seed, ids, names) = build_reference(&cfg, prompt.len());
    let store = store_from_graph(&g_seed, &ids, &names);

    let gen_cfg = GenerationConfig {
        max_new_tokens: 3,
        eos_token_ids: vec![9999],
    };
    let mut received: Vec<u32> = Vec::new();
    let generated = generate_greedy(
        &cfg,
        &store,
        &prompt,
        &gen_cfg,
        |id| format!("[{id}]"),
        &mut |t: &atenia_engine::nn::llama::GeneratedToken| {
            received.push(t.token_id);
        },
    )
    .unwrap();

    assert_eq!(received, generated);
}
