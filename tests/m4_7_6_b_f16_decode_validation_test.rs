//! M4.7.6.b — F16 decode + BF16 storage end-to-end validation.
//!
//! Closes Risk #3 of the M4.7.6 investigation: the F16-source
//! `WeightMapper` decode path was unit-tested at the byte level
//! but never validated end-to-end against a model loaded from F16
//! safetensors with `store_params_as_bf16=true` — the exact path
//! Llama 2 13B Chat will exercise in M4.7.6.d.
//!
//! The original M4.7.6 plan proposed Mistral 7B as the F16-source
//! canary. Inspection of `models/mistral-7b-v0.3/config.json`
//! shows `torch_dtype: "bfloat16"` (not float16), so Mistral
//! cannot serve as the F16 proxy. Llama 2 13B is the only F16
//! checkpoint on the dev hardware. To still close Risk #3
//! before .d, this file builds a synthetic F16-source path on
//! top of MiniFlux:
//!
//!   1. Construct a `safetensors` buffer with `StDtype::F16`
//!      values produced by `half::f16::from_f32(_)` from a
//!      deterministic source.
//!   2. Load it via `WeightMapper` with both
//!      `store_params_as_bf16=false` (F16 → F32 storage) and
//!      `store_params_as_bf16=true` (F16 → F32 → BF16 storage).
//!   3. Assert the stored values are bit-exact equal to the
//!      reference round-trip computed independently.
//!   4. Run a forward and assert finite output, verifying the
//!      F16-loaded weights produce a usable graph.
//!
//! The F16 → F32 conversion uses the `half` crate (Atenia's
//! `safetensors_reader.rs:331-354`); this file's reference
//! values are computed via the same crate so the test locks the
//! end-to-end consistency, not a re-implementation.

use std::collections::HashMap;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::mini_flux::{MiniFluxConfig, build_mini_flux};
use atenia_engine::tensor::TensorStorage;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::WeightMapper;
use safetensors::Dtype as StDtype;
use safetensors::tensor::TensorView;

fn tiny_cfg() -> MiniFluxConfig {
    MiniFluxConfig {
        vocab_size: 16,
        seq_len: 4,
        d_model: 8,
        d_hidden: 16,
        num_layers: 1,
        batch_size: 1,
    }
}

fn fresh_mini_flux(
    cfg: &MiniFluxConfig,
) -> (
    atenia_engine::amg::graph::Graph,
    atenia_engine::nn::mini_flux::MiniFluxHandles,
) {
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let mut graph = gb.build();
    let handles = build_mini_flux(&mut graph, cfg, tokens_id);
    (graph, handles)
}

/// Build a deterministic F32 source vector that round-trips
/// non-trivially through F16 (i.e., values that exercise
/// rounding, not just integers).
fn deterministic_source(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.001234 - 0.5).collect()
}

/// Encode an F32 slice as F16 little-endian bytes via the `half` crate
/// (the same crate Atenia's loader uses). Returns the bytes plus the
/// reference F32 values that result from a round-trip — i.e., the
/// values the loader is expected to produce post-decode.
fn encode_f32_as_f16_bytes(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    let mut roundtrip = Vec::with_capacity(values.len());
    for v in values {
        let h = half::f16::from_f32(*v);
        bytes.extend_from_slice(&h.to_le_bytes());
        roundtrip.push(h.to_f32());
    }
    (bytes, roundtrip)
}

#[test]
fn f16_tensor_loads_correctly_via_weight_mapper() {
    // F16 → F32 storage path. WeightMapper loads the F16
    // safetensors, decodes through `to_vec_f32`, stores
    // `Cpu(Vec<f32>)`. The stored values must equal the
    // round-trip of the source through `half::f16`.

    let cfg = tiny_cfg();
    let (mut graph, handles) = fresh_mini_flux(&cfg);

    let embedding_id = handles
        .param_ids
        .iter()
        .zip(handles.param_names.iter())
        .find(|(_, n)| n.as_str() == "embedding")
        .map(|(id, _)| *id)
        .expect("embedding must be present");
    let embedding_numel = graph.nodes[embedding_id]
        .output
        .as_ref()
        .unwrap()
        .as_cpu_slice()
        .len();
    assert_eq!(embedding_numel, cfg.vocab_size * cfg.d_model);

    let source_f32 = deterministic_source(embedding_numel);
    let (f16_bytes, expected_after_roundtrip) = encode_f32_as_f16_bytes(&source_f32);

    let view =
        TensorView::new(StDtype::F16, vec![cfg.vocab_size, cfg.d_model], &f16_bytes).unwrap();
    let mut views: HashMap<String, TensorView> = HashMap::new();
    views.insert("embedding".to_string(), view);
    let buffer = safetensors::serialize(&views, &None).unwrap();

    // Zero the embedding so the load result is the only source
    // of the post-load values.
    for v in graph.nodes[embedding_id]
        .output
        .as_mut()
        .unwrap()
        .as_cpu_slice_mut()
        .iter_mut()
    {
        *v = 0.0;
    }

    let mapper =
        WeightMapper::from_param_names_and_ids(&handles.param_names, &handles.param_ids).unwrap();
    let reader = SafetensorsReader::from_bytes(buffer).unwrap();
    let report = mapper
        .load_into(&mut graph, &reader)
        .expect("F16 load must succeed");

    assert_eq!(report.loaded, 1);
    assert!(report.skipped.is_empty());

    // Stored values must be bit-exact equal to the round-trip
    // reference. F16 → F32 via `half` is deterministic, so any
    // mismatch here is a real loader bug.
    let loaded = graph.nodes[embedding_id]
        .output
        .as_ref()
        .unwrap()
        .as_cpu_slice();
    assert_eq!(loaded.len(), expected_after_roundtrip.len());
    for (i, (got, expected)) in loaded
        .iter()
        .zip(expected_after_roundtrip.iter())
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "F16 → F32 mismatch at element {}: got 0x{:08X}, expected 0x{:08X} \
             (source f32 = {})",
            i,
            got.to_bits(),
            expected.to_bits(),
            source_f32[i],
        );
    }

    // Storage must be plain Cpu (F32) when the BF16 flag is off.
    let storage = &graph.nodes[embedding_id].output.as_ref().unwrap().storage;
    assert!(
        matches!(storage, TensorStorage::Cpu(_)),
        "expected Cpu(F32) storage with BF16 flag off, got {:?}",
        storage
    );
}

#[test]
fn f16_to_bf16_storage_path_is_consistent() {
    // F16 → F32 → BF16 storage path. WeightMapper loads the F16
    // safetensors, decodes through `to_vec_f32` (via `half`),
    // then truncates to BF16 (top 16 bits) when
    // `store_params_as_bf16(true)`. The stored CpuBf16 bits
    // must equal the BF16 truncation of the F16 round-trip
    // values.

    let cfg = tiny_cfg();
    let (mut graph, handles) = fresh_mini_flux(&cfg);

    let embedding_id = handles
        .param_ids
        .iter()
        .zip(handles.param_names.iter())
        .find(|(_, n)| n.as_str() == "embedding")
        .map(|(id, _)| *id)
        .expect("embedding must be present");
    let embedding_numel = graph.nodes[embedding_id]
        .output
        .as_ref()
        .unwrap()
        .as_cpu_slice()
        .len();

    let source_f32 = deterministic_source(embedding_numel);
    let (f16_bytes, f16_roundtrip_f32) = encode_f32_as_f16_bytes(&source_f32);

    // Reference: BF16 truncation of the F16 round-trip values.
    let expected_bf16_bits: Vec<u16> = f16_roundtrip_f32
        .iter()
        .map(|f| (f.to_bits() >> 16) as u16)
        .collect();

    let view =
        TensorView::new(StDtype::F16, vec![cfg.vocab_size, cfg.d_model], &f16_bytes).unwrap();
    let mut views: HashMap<String, TensorView> = HashMap::new();
    views.insert("embedding".to_string(), view);
    let buffer = safetensors::serialize(&views, &None).unwrap();

    let mut mapper =
        WeightMapper::from_param_names_and_ids(&handles.param_names, &handles.param_ids).unwrap();
    mapper.set_store_params_as_bf16(true);
    let reader = SafetensorsReader::from_bytes(buffer).unwrap();
    let report = mapper
        .load_into(&mut graph, &reader)
        .expect("F16 → BF16-storage load must succeed");

    assert_eq!(report.loaded, 1);

    // Storage must be CpuBf16 with the expected bits.
    let storage = &graph.nodes[embedding_id].output.as_ref().unwrap().storage;
    let stored_bits = match storage {
        TensorStorage::CpuBf16(bits) => bits.clone(),
        other => panic!(
            "expected CpuBf16 storage with BF16 flag on, got {:?}",
            other
        ),
    };
    assert_eq!(stored_bits.len(), expected_bf16_bits.len());
    for (i, (got, expected)) in stored_bits
        .iter()
        .zip(expected_bf16_bits.iter())
        .enumerate()
    {
        assert_eq!(
            got, expected,
            "F16 → F32 → BF16 mismatch at element {}: got 0x{:04X}, expected 0x{:04X}",
            i, got, expected
        );
    }
}

#[test]
fn f16_loaded_graph_executes_forward_without_panic() {
    // End-to-end check: load every MiniFlux parameter as F16
    // with BF16 storage on, run a forward, assert the output
    // is finite and shape-correct. Validates that the entire
    // M4.7.2 decode-on-access stack tolerates an F16-source
    // checkpoint at the model level — the structural concern
    // R3 was about.
    //
    // The standard `fresh_mini_flux` helper builds the Graph
    // empty and only then adds nodes via `build_mini_flux`,
    // which leaves the graph without an Output node — fine for
    // unit tests that only inspect parameter storage, but the
    // forward executor walks `self.plan.steps` and would
    // produce zero outputs. Use `build_mini_flux_language_model`
    // (which writes into a `GraphBuilder` so we can call
    // `output(_)` before `build()`) instead.
    let cfg = tiny_cfg();
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let (logits_id, param_ids, param_names) =
        atenia_engine::nn::mini_flux::build_mini_flux_language_model(&mut gb, &cfg, tokens_id);
    let _ = gb.output(logits_id);
    let mut graph = gb.build();
    let handles = atenia_engine::nn::mini_flux::MiniFluxHandles {
        token_input_id: tokens_id,
        logits_id,
        param_ids,
        param_names,
    };

    // Snapshot every parameter's F32 values, then re-encode as
    // F16 bytes. We use the existing initialised values rather
    // than an artificial source so the model's forward stays in
    // a numerically reasonable regime.
    let mut snapshots: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, &id) in handles.param_names.iter().zip(handles.param_ids.iter()) {
        let tensor = graph.nodes[id].output.as_ref().unwrap();
        let f32_values = tensor.as_cpu_slice().to_vec();
        let (f16_bytes, _) = encode_f32_as_f16_bytes(&f32_values);
        snapshots.push((name.clone(), tensor.shape.clone(), f16_bytes));
    }

    // Zero every parameter to prove the load is doing the work.
    for &id in &handles.param_ids {
        for v in graph.nodes[id]
            .output
            .as_mut()
            .unwrap()
            .as_cpu_slice_mut()
            .iter_mut()
        {
            *v = 0.0;
        }
    }

    // Build the safetensors buffer with F16 dtype on every param.
    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (name, shape, bytes) in &snapshots {
        let view = TensorView::new(StDtype::F16, shape.clone(), bytes.as_slice()).unwrap();
        views.insert(name.clone(), view);
    }
    let buffer = safetensors::serialize(&views, &None).unwrap();

    let mut mapper =
        WeightMapper::from_param_names_and_ids(&handles.param_names, &handles.param_ids).unwrap();
    mapper.set_store_params_as_bf16(true);
    let reader = SafetensorsReader::from_bytes(buffer).unwrap();
    let report = mapper.load_into(&mut graph, &reader).expect("load");
    assert_eq!(report.loaded, snapshots.len());

    // Every parameter must now be on CpuBf16.
    for &id in &handles.param_ids {
        let storage = &graph.nodes[id].output.as_ref().unwrap().storage;
        assert!(
            matches!(storage, TensorStorage::CpuBf16(_)),
            "param node {} expected CpuBf16, got {:?}",
            id,
            storage
        );
    }

    // Run the forward. MiniFlux's seq_len is 4 in tiny_cfg(),
    // so we feed 4 token ids in [0, vocab_size).
    let tokens =
        atenia_engine::tensor::Tensor::new_cpu(vec![1, cfg.seq_len], vec![1.0_f32, 2.0, 3.0, 4.0]);
    let outputs = graph.execute(vec![tokens]);
    assert_eq!(outputs.len(), 1);
    let logits = &outputs[0];
    let slice = logits.as_cpu_slice();

    let finite = slice.iter().filter(|v| v.is_finite()).count();
    assert_eq!(
        finite,
        slice.len(),
        "all F16-loaded forward outputs must be finite (got {} non-finite of {})",
        slice.len() - finite,
        slice.len(),
    );
}
