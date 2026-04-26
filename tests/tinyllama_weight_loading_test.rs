//! Tests for the M4.5-b1 `LoadTransform` pipeline and the
//! TinyLlama-specific weight-mapper helper.

use std::collections::HashMap;
use std::path::Path;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::tinyllama::weight_loading::compute_transforms_for_name;
use atenia_engine::nn::tinyllama::{tinyllama_weight_mapper, TinyLlamaConfig};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::{LoadTransform, WeightMapper};
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

const TINYLLAMA_CONFIG_JSON: &str = r#"{
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

// ---------------------------------------------------------------------------
// Helper: build a tiny safetensors buffer holding one F32 tensor with a
// given name, shape, and values.
// ---------------------------------------------------------------------------
fn build_one_tensor_safetensors(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let view = TensorView::new(StDtype::F32, shape.to_vec(), &bytes)
        .expect("TensorView construction");
    let mut views: HashMap<String, TensorView> = HashMap::new();
    views.insert(name.to_string(), view);
    safetensors::serialize(&views, &None).expect("safetensors serialize")
}

// ---------------------------------------------------------------------------
// Helper: build a 1-node graph with a Parameter at `expected_shape`,
// then run the configured transform pipeline through `load_into`.
// Returns the loaded values flat.
// ---------------------------------------------------------------------------
fn run_pipeline_on_synthetic(
    file_shape: Vec<usize>,
    file_values: Vec<f32>,
    expected_shape: Vec<usize>,
    transforms: Vec<LoadTransform>,
) -> Vec<f32> {
    let buf = build_one_tensor_safetensors("w", &file_shape, &file_values);
    let reader = SafetensorsReader::from_bytes(buf).expect("reader");

    // Materialize a parameter node with the expected post-transform shape.
    let mut gb = GraphBuilder::new();
    let pid = gb.parameter(Tensor::new_cpu(
        expected_shape.clone(),
        vec![0.0_f32; expected_shape.iter().product()],
    ));
    let _ = gb.output(pid);
    let mut graph = gb.build();

    let mut mapper =
        WeightMapper::from_param_names_and_ids(&[String::from("w")], &[pid]).unwrap();
    if !transforms.is_empty() {
        mapper.set_transforms("w", transforms).unwrap();
    }

    let report = mapper.load_into(&mut graph, &reader).expect("load_into");
    assert_eq!(report.loaded, 1, "should have loaded exactly one tensor");
    assert!(report.skipped.is_empty(), "no skipped expected: {:?}", report.skipped);
    assert!(report.missing.is_empty(), "no missing expected");

    graph.nodes[pid]
        .output
        .as_ref()
        .unwrap()
        .as_cpu_slice()
        .to_vec()
}

// ---------------------------------------------------------------------------
// Test 1 — Transpose-on-load
// ---------------------------------------------------------------------------
#[test]
fn transpose_on_load_produces_correct_shape_and_values() {
    // File: [3, 4] row-major  0..11
    let file = (0..12).map(|i| i as f32).collect::<Vec<_>>();
    let got = run_pipeline_on_synthetic(
        vec![3, 4],
        file,
        vec![4, 3],
        vec![LoadTransform::Transpose2D],
    );
    // Expected: [4, 3] where row r col c equals file[c, r]
    // = c * 4 + r interpreted as index into the original [3,4]? No:
    // transpose semantics out[c, r] == in[r, c]
    // Flat row-major out: out[c*3 + r] = in[r*4 + c]
    let mut expected = vec![0.0_f32; 12];
    for c in 0..4 {
        for r in 0..3 {
            expected[c * 3 + r] = (r * 4 + c) as f32;
        }
    }
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// Test 2 — Tile factor 2 with explicit values
// ---------------------------------------------------------------------------
#[test]
fn tile_grouped_dim_factor_2_replicates_each_block() {
    // File: [4, 3] = two groups of two rows along dim 0:
    //   group 0: rows [0,1,2] and [3,4,5]
    //   group 1: rows [6,7,8] and [9,10,11]
    // After TileGroupedDim { dim: 0, group_size: 2, repeats: 2 }:
    //   shape [8, 3], group 0 appears at rows 0..2 and 2..4, group 1 at 4..6 and 6..8.
    let file: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let got = run_pipeline_on_synthetic(
        vec![4, 3],
        file,
        vec![8, 3],
        vec![LoadTransform::TileGroupedDim {
            dim: 0,
            group_size: 2,
            repeats: 2,
        }],
    );
    let expected: Vec<f32> = vec![
        0.0, 1.0, 2.0, // group 0 row 0
        3.0, 4.0, 5.0, // group 0 row 1
        0.0, 1.0, 2.0, // group 0 row 0 (repeat)
        3.0, 4.0, 5.0, // group 0 row 1 (repeat)
        6.0, 7.0, 8.0, // group 1 row 0
        9.0, 10.0, 11.0, // group 1 row 1
        6.0, 7.0, 8.0, // group 1 row 0 (repeat)
        9.0, 10.0, 11.0, // group 1 row 1 (repeat)
    ];
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// Test 3 — Scale on load
// ---------------------------------------------------------------------------
#[test]
fn scale_on_load_multiplies_every_element() {
    let file: Vec<f32> = vec![2.0, -4.0, 6.0, 8.0];
    let got = run_pipeline_on_synthetic(
        vec![2, 2],
        file.clone(),
        vec![2, 2],
        vec![LoadTransform::Scale { factor: 0.5 }],
    );
    let expected: Vec<f32> = file.iter().map(|v| v * 0.5).collect();
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// Test 4 — Composite transform (the K_proj-style pipeline)
// ---------------------------------------------------------------------------
#[test]
fn composite_tile_then_transpose_then_scale_matches_manual_computation() {
    // Mimic K_proj at small scale: HF shape [n_kv*head_dim, hidden] = [4, 6],
    // with head_dim=2, n_kv=2, kv_groups=2, attention_scale=0.5.
    // Pipeline: Tile (dim 0, group 2, repeats 2) → Transpose2D → Scale 0.5.
    // Final shape: [hidden, n_q*head_dim] = [6, 8].
    let file: Vec<f32> = (0..24).map(|i| i as f32).collect();

    let transforms = vec![
        LoadTransform::TileGroupedDim {
            dim: 0,
            group_size: 2,
            repeats: 2,
        },
        LoadTransform::Transpose2D,
        LoadTransform::Scale { factor: 0.5 },
    ];

    let got = run_pipeline_on_synthetic(vec![4, 6], file.clone(), vec![6, 8], transforms);

    // Compute expected step by step.
    // Step 1 — tile: [4, 6] → [8, 6]
    let mut tiled = vec![0.0_f32; 8 * 6];
    let blocks: [&[f32]; 2] = [
        &file[0..12], // rows 0..2 (group 0)
        &file[12..24], // rows 2..4 (group 1)
    ];
    for (g, blk) in blocks.iter().enumerate() {
        for r in 0..2 {
            // 2 repeats per group
            let dst = (g * 2 + r) * 12;
            tiled[dst..dst + 12].copy_from_slice(blk);
        }
    }
    // Step 2 — transpose [8, 6] → [6, 8]
    let mut transposed = vec![0.0_f32; 48];
    for r in 0..8 {
        for c in 0..6 {
            transposed[c * 8 + r] = tiled[r * 6 + c];
        }
    }
    // Step 3 — scale 0.5
    let expected: Vec<f32> = transposed.iter().map(|v| v * 0.5).collect();

    assert_eq!(got.len(), expected.len());
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-7,
            "composite mismatch at {}: got {} expected {}",
            i,
            g,
            e
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5 — TinyLlama helper produces expected per-name transforms
// ---------------------------------------------------------------------------
#[test]
fn tinyllama_weight_mapper_dispatches_each_name_correctly() {
    let cfg = TinyLlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
    let head_dim = cfg.head_dim();
    let kv_groups = cfg.kv_groups();
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let hidden = cfg.hidden_size;
    let cases: &[(&str, Vec<LoadTransform>)] = &[
        ("model.embed_tokens.weight", vec![]),
        (
            "model.norm.weight",
            vec![LoadTransform::Reshape {
                target: vec![1, 1, hidden],
            }],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![LoadTransform::Reshape {
                target: vec![1, 1, hidden],
            }],
        ),
        (
            "model.layers.5.post_attention_layernorm.weight",
            vec![LoadTransform::Reshape {
                target: vec![1, 1, hidden],
            }],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![LoadTransform::Transpose2D],
        ),
        (
            "model.layers.7.self_attn.o_proj.weight",
            vec![LoadTransform::Transpose2D],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![
                LoadTransform::TileGroupedDim {
                    dim: 0,
                    group_size: head_dim,
                    repeats: kv_groups,
                },
                LoadTransform::Transpose2D,
                LoadTransform::Scale { factor: scale },
            ],
        ),
        (
            "model.layers.21.self_attn.v_proj.weight",
            vec![
                LoadTransform::TileGroupedDim {
                    dim: 0,
                    group_size: head_dim,
                    repeats: kv_groups,
                },
                LoadTransform::Transpose2D,
            ],
        ),
        (
            "model.layers.3.mlp.gate_proj.weight",
            vec![LoadTransform::Transpose2D],
        ),
        (
            "model.layers.3.mlp.up_proj.weight",
            vec![LoadTransform::Transpose2D],
        ),
        (
            "model.layers.3.mlp.down_proj.weight",
            vec![LoadTransform::Transpose2D],
        ),
        ("lm_head.weight", vec![LoadTransform::Transpose2D]),
    ];

    for (name, expected) in cases {
        let got = compute_transforms_for_name(name, hidden, head_dim, kv_groups, scale);
        assert_eq!(&got, expected, "transform list for `{}` mismatch", name);
    }

    // And also: the full mapper builder accepts a synthetic
    // names/ids vec and attaches transforms correctly.
    let names: Vec<String> = cases.iter().map(|c| c.0.to_string()).collect();
    let ids: Vec<usize> = (0..cases.len()).collect();
    let mapper = tinyllama_weight_mapper(&cfg, &names, &ids).unwrap();
    for (name, expected) in cases {
        let entry = mapper.get_mapping(name).expect("entry must exist");
        assert_eq!(entry.transforms, *expected);
    }
}

// ---------------------------------------------------------------------------
// Test 6 — set_transforms rejects unknown names
// ---------------------------------------------------------------------------
#[test]
fn set_transforms_on_unknown_name_returns_error() {
    let mut mapper = WeightMapper::from_param_names_and_ids(&[], &[]).unwrap();
    let err = mapper
        .set_transforms("does_not_exist", vec![LoadTransform::Transpose2D])
        .expect_err("must fail");
    let msg = format!("{:?}", err);
    assert!(msg.contains("not in the mapper"), "got: {}", msg);
}

// ---------------------------------------------------------------------------
// Test 7 — empty transforms behaves identically to M4-c (regression)
// ---------------------------------------------------------------------------
#[test]
fn empty_transforms_path_is_bit_exact_with_direct_copy() {
    let file: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let got = run_pipeline_on_synthetic(vec![2, 3], file.clone(), vec![2, 3], vec![]);
    assert_eq!(got, file);
}

// ---------------------------------------------------------------------------
// Test 8 — post-transform shape mismatch is reported clearly
// ---------------------------------------------------------------------------
#[test]
fn post_transform_shape_mismatch_surfaces_error() {
    let file: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let buf = build_one_tensor_safetensors("w", &[2, 3], &file);
    let reader = SafetensorsReader::from_bytes(buf).unwrap();

    // Graph parameter has the WRONG shape for the post-transpose result.
    // Post-transpose of [2, 3] is [3, 2]; the graph is set to [2, 3].
    let mut gb = GraphBuilder::new();
    let pid = gb.parameter(Tensor::new_cpu(vec![2, 3], vec![0.0_f32; 6]));
    let _ = gb.output(pid);
    let mut graph = gb.build();

    let mut mapper =
        WeightMapper::from_param_names_and_ids(&[String::from("w")], &[pid]).unwrap();
    mapper.set_transforms("w", vec![LoadTransform::Transpose2D]).unwrap();

    let err = mapper
        .load_into(&mut graph, &reader)
        .expect_err("expected shape mismatch");
    let msg = format!("{:?}", err);
    assert!(msg.contains("ShapeMismatch"), "got error: {}", msg);
    // expected = graph shape = [2, 3]; actual = post-transform = [3, 2]
    assert!(msg.contains("[3, 2]"), "actual shape should appear: {}", msg);
}

// ---------------------------------------------------------------------------
// Reshape transform (M4.5-b1 extension)
// ---------------------------------------------------------------------------
#[test]
fn reshape_transform_changes_shape_preserves_data() {
    let file: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let got = run_pipeline_on_synthetic(
        vec![4],
        file.clone(),
        vec![1, 1, 4],
        vec![LoadTransform::Reshape {
            target: vec![1, 1, 4],
        }],
    );
    assert_eq!(got, file, "Reshape must preserve data byte-for-byte");
}

#[test]
fn reshape_transform_rejects_numel_mismatch() {
    let buf = build_one_tensor_safetensors("w", &[4], &[1.0, 2.0, 3.0, 4.0]);
    let reader = SafetensorsReader::from_bytes(buf).unwrap();

    let mut gb = GraphBuilder::new();
    let pid = gb.parameter(Tensor::new_cpu(vec![1, 1, 5], vec![0.0_f32; 5]));
    let _ = gb.output(pid);
    let mut graph = gb.build();

    let mut mapper =
        WeightMapper::from_param_names_and_ids(&[String::from("w")], &[pid]).unwrap();
    mapper
        .set_transforms(
            "w",
            vec![LoadTransform::Reshape {
                target: vec![1, 1, 5],
            }],
        )
        .unwrap();

    let err = mapper
        .load_into(&mut graph, &reader)
        .expect_err("numel mismatch must fail");
    let msg = format!("{:?}", err);
    assert!(msg.contains("Reshape target"), "got: {}", msg);
    assert!(msg.contains("numel 5"), "got: {}", msg);
}

// ---------------------------------------------------------------------------
// Test 9 — load real TinyLlama q_proj weight from disk (#[ignore])
// ---------------------------------------------------------------------------
#[test]
#[ignore]
fn load_real_tinyllama_layer0_q_proj_weight_via_pipeline() {
    let path = Path::new("models/tinyllama-1.1b/model.safetensors");
    let reader = SafetensorsReader::open(path).expect("open real safetensors");

    let name = "model.layers.0.self_attn.q_proj.weight";
    assert!(
        reader.iter().any(|e| e.name == name),
        "real file must contain `{}`",
        name
    );

    // Build a minimal graph: one Parameter sized [hidden, hidden] (post-transpose).
    let hidden = 2048_usize;
    let mut gb = GraphBuilder::new();
    let pid = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], vec![0.0_f32; hidden * hidden]));
    let _ = gb.output(pid);
    let mut graph = gb.build();

    let mut mapper =
        WeightMapper::from_param_names_and_ids(&[name.to_string()], &[pid]).unwrap();
    mapper
        .set_transforms(name, vec![LoadTransform::Transpose2D])
        .unwrap();

    let report = mapper.load_into(&mut graph, &reader).expect("load_into");
    assert_eq!(report.loaded, 1);

    let values = graph.nodes[pid].output.as_ref().unwrap().as_cpu_slice();
    assert_eq!(values.len(), hidden * hidden);

    // Sanity: not all zeros, all finite, magnitudes plausible for a
    // trained Linear weight.
    let mut nonzero = 0_usize;
    for &v in values {
        assert!(v.is_finite(), "non-finite weight value: {}", v);
        if v != 0.0 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero > values.len() / 2,
        "expected most weights to be non-zero, got {}/{}",
        nonzero,
        values.len()
    );
}
