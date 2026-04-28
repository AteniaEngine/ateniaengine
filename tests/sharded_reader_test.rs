//! Integration tests for `ShardedSafetensorsReader` (M4.7.1.b).
//!
//! Builds synthetic 2-shard checkpoints in a temp directory, loads
//! them through the sharded driver, and verifies that the resulting
//! graph state is bit-identical to a single-file load of the same
//! tensors. No real model required.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::sharded_reader::ShardedSafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::WeightMapper;

use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

/// Helper: per-test temp dir under `std::env::temp_dir()`. Uses a
/// time-derived suffix to avoid collisions when several tests run
/// in parallel.
fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("atenia_{}_{}_{}", prefix, std::process::id(), nanos));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Helper: serialise a list of (name, shape, values) entries into
/// safetensors bytes that `SafetensorsReader::from_bytes` can parse.
fn serialize_tensors(entries: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    // Owned byte buffers must outlive the views.
    let raw: Vec<(String, Vec<usize>, Vec<u8>)> = entries
        .iter()
        .map(|(name, shape, values)| {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            (name.clone(), shape.clone(), bytes)
        })
        .collect();

    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (name, shape, bytes) in &raw {
        let view = TensorView::new(StDtype::F32, shape.clone(), bytes.as_slice())
            .expect("TensorView::new");
        views.insert(name.clone(), view);
    }
    safetensors::serialize(&views, &None).expect("safetensors::serialize")
}

/// Build a tiny graph with three named parameters of known shapes.
/// Returns the graph, the names, and the node ids.
fn build_three_param_graph() -> (
    atenia_engine::amg::graph::Graph,
    Vec<String>,
    Vec<usize>,
) {
    let mut gb = GraphBuilder::new();

    // Three deterministically-sized tensors, distinct contents.
    let t_a = Tensor::new_cpu(vec![2, 3], vec![0.0_f32; 6]);
    let t_b = Tensor::new_cpu(vec![4], vec![0.0_f32; 4]);
    let t_c = Tensor::new_cpu(vec![3, 2], vec![0.0_f32; 6]);

    let id_a = gb.parameter(t_a);
    let id_b = gb.parameter(t_b);
    let id_c = gb.parameter(t_c);

    let graph = gb.build();
    (
        graph,
        vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
        vec![id_a, id_b, id_c],
    )
}

#[test]
fn sharded_load_two_shards_matches_single_file_load_bit_exact() {
    // Reference values for each tensor — the data the loader must
    // reproduce in the graph after loading.
    let alpha_values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let beta_values: Vec<f32> = vec![-1.5, -2.5, -3.5, -4.5];
    let gamma_values: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

    // ---- Build a 2-shard checkpoint on disk ----
    let model_dir = unique_temp_dir("sharded_two_shards");

    // Shard 1: alpha + beta. Shard 2: gamma.
    let shard1_path = model_dir.join("model-00001-of-00002.safetensors");
    let shard2_path = model_dir.join("model-00002-of-00002.safetensors");
    fs::write(
        &shard1_path,
        serialize_tensors(&[
            ("alpha".to_string(), vec![2, 3], alpha_values.clone()),
            ("beta".to_string(), vec![4], beta_values.clone()),
        ]),
    )
    .unwrap();
    fs::write(
        &shard2_path,
        serialize_tensors(&[("gamma".to_string(), vec![3, 2], gamma_values.clone())]),
    )
    .unwrap();

    // Write the index file.
    let index_path = model_dir.join("model.safetensors.index.json");
    fs::write(
        &index_path,
        r#"{
            "metadata": { "total_size": 64 },
            "weight_map": {
                "alpha": "model-00001-of-00002.safetensors",
                "beta":  "model-00001-of-00002.safetensors",
                "gamma": "model-00002-of-00002.safetensors"
            }
        }"#,
    )
    .unwrap();

    // ---- Path A: load via ShardedSafetensorsReader ----
    let (mut graph_sharded, names, ids) = build_three_param_graph();
    let mapper_sharded =
        WeightMapper::from_param_names_and_ids(&names, &ids).expect("mapper builds");

    let sharded_reader = ShardedSafetensorsReader::open(&index_path).expect("open sharded");
    assert_eq!(sharded_reader.shard_count(), 2);
    assert_eq!(sharded_reader.tensor_count(), 3);

    let report = sharded_reader
        .load_into(&mut graph_sharded, &mapper_sharded)
        .expect("sharded load_into");

    assert_eq!(report.loaded, 3, "all three tensors loaded");
    assert!(report.skipped.is_empty(), "skipped: {:?}", report.skipped);
    assert!(report.missing.is_empty(), "missing: {:?}", report.missing);

    // ---- Path B: load via single-file SafetensorsReader on a
    //              merged synthetic file containing all three. ----
    let merged = serialize_tensors(&[
        ("alpha".to_string(), vec![2, 3], alpha_values.clone()),
        ("beta".to_string(), vec![4], beta_values.clone()),
        ("gamma".to_string(), vec![3, 2], gamma_values.clone()),
    ]);
    let single_reader = SafetensorsReader::from_bytes(merged).expect("single reader");

    let (mut graph_single, names_b, ids_b) = build_three_param_graph();
    let mapper_single =
        WeightMapper::from_param_names_and_ids(&names_b, &ids_b).expect("mapper builds");
    let single_report = mapper_single
        .load_into(&mut graph_single, &single_reader)
        .expect("single-file load");
    assert_eq!(single_report.loaded, 3);
    assert!(single_report.missing.is_empty());

    // ---- Bit-exact equivalence between sharded and single-file
    //      loads of the same tensor set. ----
    for (name, &id) in names.iter().zip(ids.iter()) {
        let sharded_slice = graph_sharded.nodes[id]
            .output
            .as_ref()
            .unwrap()
            .as_cpu_slice();
        let single_slice = graph_single.nodes[id]
            .output
            .as_ref()
            .unwrap()
            .as_cpu_slice();
        assert_eq!(
            sharded_slice, single_slice,
            "mismatch on `{}` between sharded and single-file loads",
            name
        );
    }

    // Cleanup
    let _ = fs::remove_dir_all(&model_dir);
}

#[test]
fn sharded_load_skipped_and_missing_aggregate_correctly() {
    // Build a checkpoint with one extra tensor (`extra`) that the
    // mapper does NOT register, and miss one mapper name that no
    // shard provides (`gamma`).
    let alpha_values: Vec<f32> = vec![1.0; 6];
    let extra_values: Vec<f32> = vec![9.0; 8];

    let model_dir = unique_temp_dir("sharded_skipped_missing");

    let shard1_path = model_dir.join("model-00001-of-00002.safetensors");
    let shard2_path = model_dir.join("model-00002-of-00002.safetensors");
    fs::write(
        &shard1_path,
        serialize_tensors(&[("alpha".to_string(), vec![2, 3], alpha_values)]),
    )
    .unwrap();
    fs::write(
        &shard2_path,
        serialize_tensors(&[("extra".to_string(), vec![2, 4], extra_values)]),
    )
    .unwrap();

    let index_path = model_dir.join("model.safetensors.index.json");
    fs::write(
        &index_path,
        r#"{
            "weight_map": {
                "alpha": "model-00001-of-00002.safetensors",
                "extra": "model-00002-of-00002.safetensors"
            }
        }"#,
    )
    .unwrap();

    let (mut graph, names, ids) = build_three_param_graph();
    let mapper = WeightMapper::from_param_names_and_ids(&names, &ids).expect("mapper builds");

    let sharded_reader = ShardedSafetensorsReader::open(&index_path).expect("open sharded");
    let report = sharded_reader
        .load_into(&mut graph, &mapper)
        .expect("sharded load_into");

    // alpha loads, extra is skipped (in shard 2, not in mapper),
    // beta and gamma are missing (in mapper, in no shard).
    assert_eq!(report.loaded, 1, "only alpha is in both file and mapper");
    assert_eq!(report.skipped, vec!["extra".to_string()]);

    let mut missing_sorted = report.missing.clone();
    missing_sorted.sort();
    assert_eq!(
        missing_sorted,
        vec!["beta".to_string(), "gamma".to_string()],
        "beta and gamma must be reported as missing exactly once"
    );

    let _ = fs::remove_dir_all(&model_dir);
}

#[test]
fn sharded_reader_index_accessors_work() {
    let model_dir = unique_temp_dir("sharded_accessors");
    let index_path = model_dir.join("model.safetensors.index.json");
    fs::write(
        &index_path,
        r#"{
            "metadata": { "total_size": 1234 },
            "weight_map": {
                "x": "model-00001-of-00002.safetensors",
                "y": "model-00002-of-00002.safetensors"
            }
        }"#,
    )
    .unwrap();

    let reader = ShardedSafetensorsReader::open(&index_path).expect("open");
    assert_eq!(reader.shard_count(), 2);
    assert_eq!(reader.tensor_count(), 2);
    assert_eq!(reader.index().total_size, 1234);

    let names = reader.shard_filenames();
    assert_eq!(names.len(), 2);
    assert!(names[0].starts_with("model-00001"));
    assert!(names[1].starts_with("model-00002"));

    // shard_path resolves relative to the index file's directory.
    let resolved = reader.shard_path("model-00001-of-00002.safetensors");
    assert_eq!(
        resolved,
        model_dir.join("model-00001-of-00002.safetensors")
    );

    let _ = fs::remove_dir_all(&model_dir);
}
