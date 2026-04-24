//! M4-c integration test: `WeightMapper::load_into` end-to-end.
//!
//! Validates that a fresh MiniFlux graph can be populated from a
//! safetensors buffer produced by a different MiniFlux graph,
//! recovering every parameter bit-exact. Also exercises the four
//! error / report paths: shape mismatch, missing, skipped, and
//! unsupported dtype.

use std::collections::HashMap;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::mini_flux::{build_mini_flux, MiniFluxConfig};
use atenia_engine::v17::loader::loader_errors::LoaderError;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::WeightMapper;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

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

/// Helper: build a MiniFlux graph fresh (default init values) and
/// return `(graph, handles)`.
fn fresh_mini_flux(
    cfg: &MiniFluxConfig,
) -> (atenia_engine::amg::graph::Graph, atenia_engine::nn::mini_flux::MiniFluxHandles) {
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let mut graph = gb.build();
    let handles = build_mini_flux(&mut graph, cfg, tokens_id);
    (graph, handles)
}

/// Helper: serialize every parameter of `graph` (indexed by
/// `handles.param_ids` / `handles.param_names`) into a safetensors
/// byte buffer.
fn serialize_params(
    graph: &atenia_engine::amg::graph::Graph,
    handles: &atenia_engine::nn::mini_flux::MiniFluxHandles,
) -> Vec<u8> {
    let mut snapshots: Vec<(String, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(handles.param_ids.len());
    for (name, &id) in handles.param_names.iter().zip(handles.param_ids.iter()) {
        let tensor = graph.nodes[id]
            .output
            .as_ref()
            .expect("param node must have tensor");
        let mut bytes = Vec::with_capacity(tensor.as_cpu_slice().len() * 4);
        for v in tensor.as_cpu_slice() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        snapshots.push((name.clone(), tensor.shape.clone(), bytes));
    }

    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (name, shape, bytes) in &snapshots {
        let view = TensorView::new(StDtype::F32, shape.clone(), bytes.as_slice())
            .expect("TensorView construction");
        views.insert(name.clone(), view);
    }
    safetensors::serialize(&views, &None).expect("safetensors serialize")
}

#[test]
fn load_into_roundtrip_bit_exact_matches_source_graph() {
    let cfg = tiny_cfg();

    // Source graph: built once, parameters initialized deterministically
    // by `register_weight`. We treat its values as the "ground truth"
    // we want the target graph to reproduce.
    let (source_graph, source_handles) = fresh_mini_flux(&cfg);
    let buffer = serialize_params(&source_graph, &source_handles);

    // Target graph: freshly built — also deterministic, so its
    // params already equal the source's. To prove the load actually
    // wrote values (not just a no-op comparison against identical
    // deterministic init), zero every target parameter before
    // loading. After `load_into` the target's values must match the
    // source bit-exact.
    let (mut target_graph, target_handles) = fresh_mini_flux(&cfg);
    for &id in &target_handles.param_ids {
        let tensor = target_graph.nodes[id]
            .output
            .as_mut()
            .expect("param tensor present");
        for v in tensor.as_cpu_slice_mut().iter_mut() {
            *v = 0.0;
        }
    }

    let mapper = WeightMapper::from_param_names_and_ids(
        &target_handles.param_names,
        &target_handles.param_ids,
    )
    .expect("mapper builds from aligned slices");

    let reader = SafetensorsReader::from_bytes(buffer).expect("reader opens");
    let report = mapper
        .load_into(&mut target_graph, &reader)
        .expect("load_into succeeds on matched shapes");

    assert_eq!(report.loaded, target_handles.param_ids.len());
    assert!(report.skipped.is_empty(), "no extra tensors in reader");
    assert!(report.missing.is_empty(), "no missing tensors in reader");

    // Per-parameter bit-exact comparison.
    for (name, &source_id) in source_handles
        .param_names
        .iter()
        .zip(source_handles.param_ids.iter())
    {
        let target_id = target_handles
            .param_names
            .iter()
            .zip(target_handles.param_ids.iter())
            .find(|(n, _)| *n == name)
            .map(|(_, id)| *id)
            .expect("target has same param name");

        let source_tensor = source_graph.nodes[source_id].output.as_ref().unwrap();
        let target_tensor = target_graph.nodes[target_id].output.as_ref().unwrap();

        assert_eq!(source_tensor.shape, target_tensor.shape);
        let source_vals = source_tensor.as_cpu_slice();
        let target_vals = target_tensor.as_cpu_slice();
        assert_eq!(source_vals.len(), target_vals.len());

        for (i, (s, t)) in source_vals.iter().zip(target_vals.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                t.to_bits(),
                "tensor '{}' element {}: source bits=0x{:08x}, target bits=0x{:08x}",
                name,
                i,
                s.to_bits(),
                t.to_bits()
            );
        }
    }
}

#[test]
fn shape_mismatch_surfaces_actionable_error() {
    let cfg = tiny_cfg();
    let (mut graph, handles) = fresh_mini_flux(&cfg);

    // Build a safetensors buffer where `w_out` has the wrong shape.
    // The graph expects `[d_model, vocab_size] = [8, 16]`; we feed
    // it `[16, 8]` instead.
    let wrong_shape = vec![16usize, 8];
    let wrong_bytes: Vec<u8> = (0..(16 * 8))
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    let view = TensorView::new(StDtype::F32, wrong_shape.clone(), wrong_bytes.as_slice())
        .expect("TensorView");
    let mut views: HashMap<String, TensorView> = HashMap::new();
    views.insert("w_out".to_string(), view);
    let buffer = safetensors::serialize(&views, &None).unwrap();

    let mapper =
        WeightMapper::from_param_names_and_ids(&handles.param_names, &handles.param_ids)
            .unwrap();
    let reader = SafetensorsReader::from_bytes(buffer).unwrap();

    let err = mapper
        .load_into(&mut graph, &reader)
        .expect_err("shape mismatch must error");
    match err {
        LoaderError::ShapeMismatch {
            tensor_name,
            expected,
            actual,
        } => {
            assert_eq!(tensor_name, "w_out");
            assert_eq!(expected, vec![cfg.d_model, cfg.vocab_size]);
            assert_eq!(actual, wrong_shape);
        }
        other => panic!("expected ShapeMismatch, got {:?}", other),
    }
}

#[test]
fn missing_tensors_reported_not_errored() {
    let cfg = tiny_cfg();
    let (mut graph, handles) = fresh_mini_flux(&cfg);

    // Serialize a buffer containing only a subset of the params
    // (skip "embedding"). The mapper still covers every name, so
    // "embedding" must land in LoadReport.missing.
    let mut snapshots: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, &id) in handles.param_names.iter().zip(handles.param_ids.iter()) {
        if name == "embedding" {
            continue;
        }
        let tensor = graph.nodes[id].output.as_ref().unwrap();
        let mut bytes = Vec::new();
        for v in tensor.as_cpu_slice() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        snapshots.push((name.clone(), tensor.shape.clone(), bytes));
    }
    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (name, shape, bytes) in &snapshots {
        let view = TensorView::new(StDtype::F32, shape.clone(), bytes.as_slice()).unwrap();
        views.insert(name.clone(), view);
    }
    let buffer = safetensors::serialize(&views, &None).unwrap();

    let mapper =
        WeightMapper::from_param_names_and_ids(&handles.param_names, &handles.param_ids)
            .unwrap();
    let reader = SafetensorsReader::from_bytes(buffer).unwrap();
    let report = mapper.load_into(&mut graph, &reader).unwrap();

    assert_eq!(report.loaded, handles.param_ids.len() - 1);
    assert!(report.skipped.is_empty());
    assert_eq!(report.missing, vec!["embedding".to_string()]);
}

#[test]
fn extra_tensors_in_reader_reported_as_skipped() {
    let cfg = tiny_cfg();
    let (mut graph, handles) = fresh_mini_flux(&cfg);

    // Serialize every real param + one extra tensor not covered by
    // the mapper. The load must succeed; the extra tensor must land
    // in LoadReport.skipped.
    let mut snapshots: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, &id) in handles.param_names.iter().zip(handles.param_ids.iter()) {
        let tensor = graph.nodes[id].output.as_ref().unwrap();
        let mut bytes = Vec::new();
        for v in tensor.as_cpu_slice() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        snapshots.push((name.clone(), tensor.shape.clone(), bytes));
    }
    // Synthetic extra.
    let extra_bytes: Vec<u8> = (0..4).flat_map(|i| (i as f32).to_le_bytes()).collect();
    snapshots.push((
        "producer_metadata_tensor".to_string(),
        vec![4usize],
        extra_bytes,
    ));

    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (name, shape, bytes) in &snapshots {
        let view = TensorView::new(StDtype::F32, shape.clone(), bytes.as_slice()).unwrap();
        views.insert(name.clone(), view);
    }
    let buffer = safetensors::serialize(&views, &None).unwrap();

    let mapper =
        WeightMapper::from_param_names_and_ids(&handles.param_names, &handles.param_ids)
            .unwrap();
    let reader = SafetensorsReader::from_bytes(buffer).unwrap();
    let report = mapper.load_into(&mut graph, &reader).unwrap();

    assert_eq!(report.loaded, handles.param_ids.len());
    assert_eq!(report.skipped, vec!["producer_metadata_tensor".to_string()]);
    assert!(report.missing.is_empty());
}

#[test]
fn bf16_tensor_loads_correctly_after_m4d() {
    // Renamed and re-scoped from the M4-c version that asserted
    // `UnsupportedDType`. With M4-d (BF16 → F32 conversion in
    // `TensorEntry::to_vec_f32`), the mapper now accepts BF16
    // checkpoints end-to-end.
    //
    // Strategy: serialize "embedding" as BF16, load it via the
    // mapper, and verify that every value in the graph matches the
    // expected BF16-to-F32 conversion (low 16 bits zero) of the
    // original f32 source.
    let cfg = tiny_cfg();
    let (mut graph, handles) = fresh_mini_flux(&cfg);

    // Source values to encode as BF16: use the "embedding" tensor
    // that already exists in the graph, truncated to BF16 precision.
    // The load will then restore the same truncated-and-upcast
    // values, so we can assert bit-exact against that reference.
    let embedding_id = handles
        .param_ids
        .iter()
        .zip(handles.param_names.iter())
        .find(|(_, n)| n.as_str() == "embedding")
        .map(|(id, _)| *id)
        .expect("embedding must be in MiniFlux params");
    let embedding_f32 = graph.nodes[embedding_id]
        .output
        .as_ref()
        .unwrap()
        .as_cpu_slice()
        .to_vec();

    let mut bf16_bytes: Vec<u8> = Vec::with_capacity(embedding_f32.len() * 2);
    let mut expected_after_truncate: Vec<f32> = Vec::with_capacity(embedding_f32.len());
    for v in &embedding_f32 {
        let top16 = (v.to_bits() >> 16) as u16;
        bf16_bytes.extend_from_slice(&top16.to_le_bytes());
        expected_after_truncate.push(f32::from_bits((top16 as u32) << 16));
    }

    let view =
        TensorView::new(StDtype::BF16, vec![cfg.vocab_size, cfg.d_model], &bf16_bytes).unwrap();
    let mut views: HashMap<String, TensorView> = HashMap::new();
    views.insert("embedding".to_string(), view);
    let buffer = safetensors::serialize(&views, &None).unwrap();

    // Zero the embedding so a no-op load cannot accidentally pass.
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
        WeightMapper::from_param_names_and_ids(&handles.param_names, &handles.param_ids)
            .unwrap();
    let reader = SafetensorsReader::from_bytes(buffer).unwrap();
    let report = mapper
        .load_into(&mut graph, &reader)
        .expect("BF16 load must succeed after M4-d");

    assert_eq!(report.loaded, 1);
    assert!(report.skipped.is_empty());

    let loaded_values = graph.nodes[embedding_id]
        .output
        .as_ref()
        .unwrap()
        .as_cpu_slice();
    for (i, (got, expected)) in loaded_values.iter().zip(expected_after_truncate.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "embedding element {}: got 0x{:08X}, expected 0x{:08X}",
            i,
            got.to_bits(),
            expected.to_bits()
        );
    }
}
