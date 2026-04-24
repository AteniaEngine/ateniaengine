//! M4-b integration test: MiniFlux ↔ safetensors roundtrip.
//!
//! Validates that parameter values registered by `build_mini_flux`
//! can be serialized into the HuggingFace safetensors format (via
//! the official `safetensors` writer) and deserialized back through
//! Atenia's own `SafetensorsReader` (introduced in M4-a) with
//! bit-exact recovery.
//!
//! This closes the M4-b scope: loader mechanics validated
//! end-to-end without depending on any external model file. The
//! same pipeline will later drive real safetensors ingestion
//! (M4-c weight mapper + M4-d BF16/F16 conversion) and eventually
//! TinyLlama / Llama 3.2 in M4.5.

use std::collections::HashMap;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::mini_flux::{build_mini_flux, MiniFluxConfig};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn tiny_cfg() -> MiniFluxConfig {
    // Smallest config that exercises every code path in
    // `build_mini_flux_internal`: at least one block (so
    // `build_block` runs once), and vocab/d_model small enough
    // that the test is fast.
    MiniFluxConfig {
        vocab_size: 16,
        seq_len: 4,
        d_model: 8,
        d_hidden: 16,
        num_layers: 1,
        batch_size: 1,
    }
}

#[test]
fn param_names_and_ids_are_index_aligned_and_nonempty() {
    let cfg = tiny_cfg();
    // Same construction pattern the existing mini_flux_common test
    // harness uses: build an empty graph via GraphBuilder, then let
    // `build_mini_flux` attach the MiniFlux subgraph to a freshly
    // declared input node.
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let mut graph = gb.build();
    let handles = build_mini_flux(&mut graph, &cfg, tokens_id);

    assert_eq!(
        handles.param_ids.len(),
        handles.param_names.len(),
        "param_ids and param_names must be index-aligned"
    );
    assert!(
        !handles.param_ids.is_empty(),
        "MiniFlux with num_layers=1 must register at least embedding + 6 block weights + w_out"
    );

    // Sanity: expected count is 1 (embedding) + 6 per block +
    // 1 (w_out) = 8 for num_layers=1.
    assert_eq!(handles.param_ids.len(), 8);
}

#[test]
fn naming_convention_contains_expected_patterns() {
    let cfg = MiniFluxConfig {
        vocab_size: 16,
        seq_len: 4,
        d_model: 8,
        d_hidden: 16,
        num_layers: 2, // probar que layers están prefijadas por índice
        batch_size: 1,
    };
    // Same construction pattern the existing mini_flux_common test
    // harness uses: build an empty graph via GraphBuilder, then let
    // `build_mini_flux` attach the MiniFlux subgraph to a freshly
    // declared input node.
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let mut graph = gb.build();
    let handles = build_mini_flux(&mut graph, &cfg, tokens_id);

    let names: &[String] = &handles.param_names;

    assert!(
        names.iter().any(|n| n == "embedding"),
        "expected 'embedding' in param_names, got: {:?}",
        names
    );
    assert!(
        names.iter().any(|n| n == "w_out"),
        "expected 'w_out' in param_names, got: {:?}",
        names
    );
    assert!(
        names.iter().any(|n| n.starts_with("layer0_")),
        "expected at least one 'layer0_' prefix in param_names"
    );
    assert!(
        names.iter().any(|n| n.starts_with("layer1_")),
        "expected at least one 'layer1_' prefix in param_names"
    );
}

#[test]
fn roundtrip_bit_exact_via_safetensors_writer_and_reader() {
    let cfg = tiny_cfg();
    // Same construction pattern the existing mini_flux_common test
    // harness uses: build an empty graph via GraphBuilder, then let
    // `build_mini_flux` attach the MiniFlux subgraph to a freshly
    // declared input node.
    let mut gb = GraphBuilder::new();
    let tokens_id = gb.input();
    let mut graph = gb.build();
    let handles = build_mini_flux(&mut graph, &cfg, tokens_id);

    // Snapshot every parameter's values + shape from the graph
    // before serialization, so the comparison survives the
    // serialize → deserialize cycle without holding references
    // into the intermediate buffer.
    let mut originals: Vec<(String, Vec<usize>, Vec<f32>)> =
        Vec::with_capacity(handles.param_ids.len());
    for (name, &id) in handles.param_names.iter().zip(handles.param_ids.iter()) {
        let tensor = graph.nodes[id]
            .output
            .as_ref()
            .expect("parameter node must have its tensor materialized at build time");
        let shape = tensor.shape.clone();
        let values = tensor.as_cpu_slice().to_vec();
        originals.push((name.clone(), shape, values));
    }

    // Build the safetensors buffer. `TensorView::new` requires
    // the raw byte slice in little-endian order, which is the
    // in-memory layout of `&[f32]` on x86_64/aarch64 little-
    // endian hosts. Atenia targets those arches only today, so
    // `bytemuck`-style cast via `from_raw_parts` is safe; we
    // do it explicitly via `to_le_bytes` chunks to keep the
    // test endianness-portable.
    let body_bytes: Vec<Vec<u8>> = originals
        .iter()
        .map(|(_, _, values)| {
            let mut buf = Vec::with_capacity(values.len() * 4);
            for v in values {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            buf
        })
        .collect();

    let mut tensor_views: HashMap<String, TensorView> = HashMap::new();
    for ((name, shape, _), bytes) in originals.iter().zip(body_bytes.iter()) {
        let view = TensorView::new(StDtype::F32, shape.clone(), bytes.as_slice())
            .expect("TensorView construction must succeed for valid F32 data");
        tensor_views.insert(name.clone(), view);
    }

    let serialized = safetensors::serialize(&tensor_views, &None)
        .expect("safetensors writer must produce a valid buffer");

    // Deserialize via Atenia's M4-a reader and compare every
    // tensor against its original `Vec<f32>` snapshot.
    let reader =
        SafetensorsReader::from_bytes(serialized).expect("reader must open serialized buffer");

    assert_eq!(
        reader.len(),
        originals.len(),
        "reader must surface every serialized tensor"
    );

    for (name, shape, values_original) in &originals {
        let entry = reader.get(name).unwrap_or_else(|| {
            panic!("tensor '{}' missing after roundtrip", name)
        });

        assert_eq!(
            entry.shape, shape.as_slice(),
            "shape mismatch for tensor '{}': got {:?}, expected {:?}",
            name, entry.shape, shape
        );

        let values_roundtrip = entry
            .to_vec_f32()
            .unwrap_or_else(|e| panic!("to_vec_f32 failed for '{}': {:?}", name, e));

        assert_eq!(
            values_roundtrip.len(),
            values_original.len(),
            "element count mismatch for '{}'",
            name
        );

        for (i, (orig, got)) in values_original.iter().zip(values_roundtrip.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                got.to_bits(),
                "tensor '{}' element {}: original {} (bits=0x{:08x}), roundtrip {} (bits=0x{:08x})",
                name,
                i,
                orig,
                orig.to_bits(),
                got,
                got.to_bits()
            );
        }
    }
}
