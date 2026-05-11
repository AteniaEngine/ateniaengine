//! M4.7.2.b — `WeightMapper::store_params_as_bf16` integration tests.
//!
//! Verifies that:
//! 1. With `store_params_as_bf16=false` (default), `load_into`
//!    behaviour is bit-identical to the M4-c baseline. Same graph
//!    parameters end up with `TensorStorage::Cpu(Vec<f32>)`.
//! 2. With `store_params_as_bf16=true`, `load_into` produces
//!    `TensorStorage::CpuBf16(Vec<u16>)` parameters. The decoded
//!    values are bit-exact equivalent to running the load with the
//!    `ATENIA_BF16_PRECISION_FLOOR` env var on (the spike from
//!    commit `a786837`), which is the contract the M4.7.2
//!    investigation locked in.

use std::collections::HashMap;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::tensor::{Tensor, TensorStorage, f32_to_bf16_bits};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::WeightMapper;
use safetensors::Dtype as StDtype;
use safetensors::tensor::TensorView;

/// Helper: build a tiny graph with two named parameters of fixed
/// shape, return `(graph, names, ids)`.
fn build_tiny_graph() -> (atenia_engine::amg::graph::Graph, Vec<String>, Vec<usize>) {
    let mut gb = GraphBuilder::new();

    // Parameters built with zero data; the loader overwrites.
    let w_shape = vec![2_usize, 3];
    let w_id = gb.parameter(Tensor::new_cpu(w_shape, vec![0.0_f32; 6]));
    let b_shape = vec![3_usize];
    let b_id = gb.parameter(Tensor::new_cpu(b_shape, vec![0.0_f32; 3]));

    let graph = gb.build();
    (
        graph,
        vec!["weight".to_string(), "bias".to_string()],
        vec![w_id, b_id],
    )
}

/// Helper: synthesise a safetensors blob with two F32 tensors named
/// "weight" (shape `[2, 3]`) and "bias" (shape `[3]`) holding the
/// values returned by the closures.
fn build_safetensors_blob<W, B>(weight_data: W, bias_data: B) -> Vec<u8>
where
    W: Fn() -> Vec<f32>,
    B: Fn() -> Vec<f32>,
{
    let w_vals = weight_data();
    let b_vals = bias_data();

    let mut w_bytes = Vec::with_capacity(w_vals.len() * 4);
    for v in &w_vals {
        w_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut b_bytes = Vec::with_capacity(b_vals.len() * 4);
    for v in &b_vals {
        b_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let w_view = TensorView::new(StDtype::F32, vec![2, 3], &w_bytes).unwrap();
    let b_view = TensorView::new(StDtype::F32, vec![3], &b_bytes).unwrap();

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert("weight".to_string(), w_view);
    tensors.insert("bias".to_string(), b_view);

    safetensors::serialize(&tensors, &None).unwrap()
}

#[test]
fn default_load_path_remains_cpu_f32() {
    // Confirms the M4.6 backward-compatibility contract: with the
    // BF16 flag off (the default), every loaded parameter ends up
    // as `TensorStorage::Cpu(Vec<f32>)` exactly as before.
    let weight_vals = || vec![1.0_f32, -2.5, 3.125, -4.0, 0.5, 0.0];
    let bias_vals = || vec![std::f32::consts::PI, -std::f32::consts::E, 1.0];
    let blob = build_safetensors_blob(weight_vals, bias_vals);
    let reader = SafetensorsReader::from_bytes(blob).unwrap();

    let (mut graph, names, ids) = build_tiny_graph();
    let mapper = WeightMapper::from_param_names_and_ids(&names, &ids).unwrap();
    assert!(!mapper.store_params_as_bf16(), "default must be off");

    let report = mapper.load_into(&mut graph, &reader).expect("load");
    assert_eq!(report.loaded, 2);

    for &id in &ids {
        let t = graph.nodes[id].output.as_ref().unwrap();
        assert!(
            matches!(t.storage(), TensorStorage::Cpu(_)),
            "default path must produce Cpu storage"
        );
    }

    let w = graph.nodes[ids[0]].output.as_ref().unwrap();
    assert_eq!(w.as_cpu_slice(), &weight_vals()[..]);
    let b = graph.nodes[ids[1]].output.as_ref().unwrap();
    assert_eq!(b.as_cpu_slice(), &bias_vals()[..]);
}

#[test]
fn bf16_flag_produces_cpu_bf16_storage_with_truncated_values() {
    // Pi has non-zero lower-mantissa bits; loading with the BF16
    // flag must yield `CpuBf16` storage, and `copy_to_cpu_vec`
    // must return values that match the truncating BF16
    // round-trip applied to each F32 input.
    let weight_vals = || {
        vec![
            std::f32::consts::PI,
            -std::f32::consts::E,
            std::f32::consts::SQRT_2,
            1.0_f32,
            -16.0,
            0.5,
        ]
    };
    let bias_vals = || vec![std::f32::consts::LN_2, -1.5, 0.0_f32];
    let blob = build_safetensors_blob(weight_vals, bias_vals);
    let reader = SafetensorsReader::from_bytes(blob).unwrap();

    let (mut graph, names, ids) = build_tiny_graph();
    let mut mapper = WeightMapper::from_param_names_and_ids(&names, &ids).unwrap();
    mapper.set_store_params_as_bf16(true);
    assert!(mapper.store_params_as_bf16());

    let report = mapper.load_into(&mut graph, &reader).expect("load bf16");
    assert_eq!(report.loaded, 2);

    let w = graph.nodes[ids[0]].output.as_ref().unwrap();
    let b = graph.nodes[ids[1]].output.as_ref().unwrap();

    // 1. Storage variant flipped to CpuBf16.
    assert!(matches!(w.storage(), TensorStorage::CpuBf16(_)));
    assert!(matches!(b.storage(), TensorStorage::CpuBf16(_)));

    // 2. Decoded values match the truncating BF16 round-trip
    //    applied to each input element. This is the same operation
    //    the precision-floor spike (commit a786837) performs, so
    //    this is the contract.
    let expected_w: Vec<f32> = weight_vals()
        .iter()
        .map(|&v| f32::from_bits((f32_to_bf16_bits(v) as u32) << 16))
        .collect();
    let expected_b: Vec<f32> = bias_vals()
        .iter()
        .map(|&v| f32::from_bits((f32_to_bf16_bits(v) as u32) << 16))
        .collect();

    let decoded_w = w.copy_to_cpu_vec();
    let decoded_b = b.copy_to_cpu_vec();
    assert_eq!(
        decoded_w, expected_w,
        "weight decoded values must equal BF16 truncation"
    );
    assert_eq!(
        decoded_b, expected_b,
        "bias decoded values must equal BF16 truncation"
    );

    // 3. Persistent storage halved: Vec<u16> instead of Vec<f32>.
    if let TensorStorage::CpuBf16(bits) = w.storage() {
        assert_eq!(bits.len(), 6);
        // Each bit pattern is exactly the upper 16 F32 bits of the
        // post-load (== input) F32 value.
        for (i, (&got, &orig)) in bits.iter().zip(weight_vals().iter()).enumerate() {
            let expected = f32_to_bf16_bits(orig);
            assert_eq!(got, expected, "bit pattern mismatch at i={}", i);
        }
    }
}

#[test]
fn bf16_path_round_trips_bit_exact_to_precision_floor_spike() {
    // The M4.7.2 contract: native BF16 storage and the
    // ATENIA_BF16_PRECISION_FLOOR spike produce mathematically
    // identical results. Run the same load both ways and assert
    // the decoded F32 vectors are bit-equal.
    let blob = build_safetensors_blob(
        || {
            vec![
                std::f32::consts::PI,
                -1.234567,
                0.000123,
                12345.6789,
                -0.5,
                1e-7,
            ]
        },
        || vec![std::f32::consts::E, 100.0_f32, -1e-5],
    );

    // Path A: BF16 flag.
    let (mut graph_a, names_a, ids_a) = build_tiny_graph();
    let reader_a = SafetensorsReader::from_bytes(blob.clone()).unwrap();
    let mut mapper_a = WeightMapper::from_param_names_and_ids(&names_a, &ids_a).unwrap();
    mapper_a.set_store_params_as_bf16(true);
    mapper_a.load_into(&mut graph_a, &reader_a).unwrap();

    // Path B: precision-floor spike via env var, no BF16 storage.
    // Set + unset around this call to keep the env clean for
    // concurrent tests.
    let key = "ATENIA_BF16_PRECISION_FLOOR";
    // SAFETY: this test runs single-threaded within its #[test]
    // function; we restore the env var before returning. Other
    // tests in the same binary do not read this var.
    unsafe {
        std::env::set_var(key, "1");
    }
    let (mut graph_b, names_b, ids_b) = build_tiny_graph();
    let reader_b = SafetensorsReader::from_bytes(blob).unwrap();
    let mapper_b = WeightMapper::from_param_names_and_ids(&names_b, &ids_b).unwrap();
    mapper_b.load_into(&mut graph_b, &reader_b).unwrap();
    unsafe {
        std::env::remove_var(key);
    }

    // Decoded values must be bit-equal across the two paths.
    for (id_a, id_b) in ids_a.iter().zip(ids_b.iter()) {
        let a = graph_a.nodes[*id_a]
            .output
            .as_ref()
            .unwrap()
            .copy_to_cpu_vec();
        let b = graph_b.nodes[*id_b]
            .output
            .as_ref()
            .unwrap()
            .as_cpu_slice()
            .to_vec();
        assert_eq!(a.len(), b.len());
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                av.to_bits(),
                bv.to_bits(),
                "BF16 storage and precision-floor spike disagree at index {} \
                 (bf16-storage={}, spike={})",
                i,
                av,
                bv
            );
        }
    }
}
