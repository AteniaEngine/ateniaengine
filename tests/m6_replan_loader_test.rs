//! M6 replan sub-fase 2 — integration tests for
//! [`WeightMapper::load_into_with_residency_plan`].
//!
//! Three test scenarios cover the contract:
//!
//! 1. **All-Ram plan** — bit-exact result vs the classic
//!    [`WeightMapper::load_into`] path. Regression guard.
//! 2. **Mixed plan (Ram + Vram)** — verifies that the store
//!    contains `SharedParam::Cuda` for Vram-tier entries and
//!    `SharedParam::Bf16` (or F32) for Ram-tier entries, and
//!    that the materialised tensors decode bit-exactly back to
//!    the source values.
//! 3. **Vram fast-path counter** — confirms that a BF16 source
//!    with no transforms goes through the raw-bytes upload
//!    path, not the F32-transient slow path. This is the
//!    structural guarantee that no `Vec<f32>` was materialised
//!    on the host during the upload.
//!
//! The CUDA-dependent tests are marked `#[ignore]` and skip on
//! non-CUDA hosts via `cuda_available()`.

use std::collections::HashMap;
use std::sync::Mutex;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::weight_store::SharedParam;
use atenia_engine::cuda::cuda_available;
use atenia_engine::gpu::tier_plan::{plan, TensorMeta, Tier, TierPlanInput};
use atenia_engine::tensor::tensor::{f32_to_bf16_bits, Tensor};
use atenia_engine::tensor::{DType, TensorStorage};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::{
    vram_fast_path_count, vram_slow_path_count, WeightMapper,
};
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Serialises tests that snapshot global atomic counters
/// (`vram_fast_path_count`, `vram_slow_path_count`,
/// `gpu_matmul_resident_count`, etc). Without this lock, the
/// default cargo test runner interleaves tests and each one's
/// "after" snapshot observes the others' increments, producing
/// flaky counter assertions. Same pattern used in
/// `src/gpu/dispatch/hooks.rs::m6_step_2b_routing_tests`.
static COUNTER_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Build an in-memory safetensors buffer with multiple BF16
/// tensors. Each `(name, shape, f32_values)` tuple becomes one
/// BF16-encoded tensor in the file (values are quantised via
/// `f32_to_bf16_bits` before serialisation).
fn build_multi_bf16_safetensors(
    entries: &[(&str, Vec<usize>, Vec<f32>)],
) -> Vec<u8> {
    // We need each tensor's BF16 bytes to outlive the `views`
    // HashMap (the `TensorView` holds a `&[u8]` borrow). Allocate
    // owned `Vec<u8>` per entry up-front.
    let bf16_byte_buffers: Vec<Vec<u8>> = entries
        .iter()
        .map(|(_, _, vals)| {
            let mut bytes = Vec::with_capacity(vals.len() * 2);
            for &v in vals {
                let bits = f32_to_bf16_bits(v);
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
            bytes
        })
        .collect();

    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (i, (name, shape, _)) in entries.iter().enumerate() {
        let view = TensorView::new(StDtype::BF16, shape.clone(), &bf16_byte_buffers[i])
            .expect("TensorView construction");
        views.insert(name.to_string(), view);
    }

    safetensors::serialize(&views, &None).expect("safetensors serialize")
}

/// Build a graph with one parameter node per tensor name. Each
/// node is allocated as a `Cpu(Vec<f32>)` zero tensor — matches
/// the production builder's slot dtype, which the default-config
/// `WeightMapper` (no `store_params_as_bf16`) writes into via
/// `as_cpu_slice_mut`. The loader's BF16-storage toggle would
/// `set_cpu_bf16_bits` to replace the slot variant; we keep the
/// default F32 path here for test-clarity.
fn build_graph_for_entries(
    entries: &[(&str, Vec<usize>, Vec<f32>)],
) -> (atenia_engine::amg::graph::Graph, Vec<usize>, Vec<String>) {
    let mut gb = GraphBuilder::new();
    let mut param_ids = Vec::with_capacity(entries.len());
    let param_names: Vec<String> = entries.iter().map(|(n, _, _)| n.to_string()).collect();
    for (_, shape, _) in entries {
        let numel: usize = shape.iter().product();
        let pid = gb.parameter(Tensor::new_cpu(shape.clone(), vec![0.0_f32; numel]));
        param_ids.push(pid);
        let _ = gb.output(pid);
    }
    (gb.build(), param_ids, param_names)
}

// ---------------------------------------------------------------------
// Test 1 — All-Ram plan: bit-exact regression vs `load_into`
// ---------------------------------------------------------------------

#[test]
fn all_ram_plan_matches_load_into_bit_exact() {
    // Two small BF16 tensors. Plan all to Ram → result must be
    // bit-exact equivalent to the classic `load_into` path.
    let entries = vec![
        ("w1", vec![3, 4], (0..12).map(|i| (i as f32) * 0.25 - 1.5).collect::<Vec<_>>()),
        ("w2", vec![2, 5], (0..10).map(|i| ((i as f32) * 0.1).sin()).collect::<Vec<_>>()),
    ];
    let buf = build_multi_bf16_safetensors(&entries);
    let reader = SafetensorsReader::from_bytes(buf.clone()).expect("reader");

    // Build two identical graphs and mappers, run the two paths.
    let (mut graph_a, param_ids_a, param_names_a) = build_graph_for_entries(&entries);
    let (mut graph_b, _param_ids_b, _param_names_b) = build_graph_for_entries(&entries);

    let mapper_a =
        WeightMapper::from_param_names_and_ids(&param_names_a, &param_ids_a).unwrap();
    let mapper_b =
        WeightMapper::from_param_names_and_ids(&param_names_a, &param_ids_a).unwrap();

    // Path A: classic `load_into`.
    let _report_a = mapper_a.load_into(&mut graph_a, &reader).expect("load_into A");
    // Hoist into Arc-shared store (same step the production
    // pipeline does).
    let store_a = atenia_engine::amg::weight_store::WeightStore::extract_from_graph(
        &mut graph_a,
        &param_ids_a,
        &param_names_a,
    )
    .expect("extract A");

    // Path B: `load_into_with_residency_plan` with all-Ram plan.
    let metas: Vec<TensorMeta> = entries
        .iter()
        .map(|(n, s, _)| TensorMeta {
            name: n.to_string(),
            shape: s.clone(),
            dtype: DType::BF16,
        })
        .collect();
    // Force everything to Ram by giving 0 free VRAM.
    let plan_input = TierPlanInput {
        tensors: metas,
        free_vram_bytes: 0,
        free_ram_bytes: 100 * 1024 * 1024 * 1024,
        // Tiny synthetic model — far below adaptive threshold; M7.2
        // policy must keep the headroom at the 8 GiB base.
        model_total_bytes: 0,
        total_ram_bytes: 128 * 1024 * 1024 * 1024,
        kernel_dtype: DType::F32,
    };
    let p = plan(&plan_input);
    assert_eq!(p.count(Tier::Vram), 0, "plan must be all-Ram for this test");
    assert_eq!(p.count(Tier::Disk), 0);

    let reader_b = SafetensorsReader::from_bytes(buf).expect("reader");
    let (store_b, _report_b) = mapper_b
        .load_into_with_residency_plan(
            &mut graph_b,
            &reader_b,
            &p,
            &param_ids_a,
            &param_names_a,
        )
        .expect("load_into_with_residency_plan B");

    // Compare the two stores element-by-element. With the
    // default mapper (no `store_params_as_bf16`) and `Cpu` graph
    // slots, both paths produce `SharedParam::F32`. Bit-exact
    // equality of the underlying `Vec<f32>` buffers proves
    // regression-zero on the all-Ram plan.
    assert_eq!(store_a.len(), store_b.len());
    for name in &param_names_a {
        let a = store_a.get_by_name(name).unwrap();
        let b = store_b.get_by_name(name).unwrap();
        match (a, b) {
            (
                SharedParam::F32 { shape: sa, arc: aa },
                SharedParam::F32 { shape: sb, arc: ab },
            ) => {
                assert_eq!(sa, sb, "shape mismatch for '{}'", name);
                assert_eq!(aa.as_slice(), ab.as_slice(),
                    "F32 bytes mismatch for '{}'", name);
            }
            _ => panic!(
                "expected both stores to hold F32 for '{}', got A={:?} B={:?}",
                name, a, b
            ),
        }
    }
}

// ---------------------------------------------------------------------
// Test 2 — Mixed plan: SharedParam::Cuda appears for Vram entries
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn mixed_plan_produces_cuda_entries_for_vram_tier() {
    let _guard = COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    // One projection-style tensor (`w_proj.weight`) → Vram
    // candidate. One non-proj tensor → Ram by default.
    let entries = vec![
        (
            "w_proj.weight",
            vec![4, 4],
            (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect::<Vec<_>>(),
        ),
        (
            "w_other",
            vec![3, 3],
            (0..9).map(|i| (i as f32) * 0.05 + 0.2).collect::<Vec<_>>(),
        ),
    ];
    let buf = build_multi_bf16_safetensors(&entries);
    let reader = SafetensorsReader::from_bytes(buf).expect("reader");

    let (mut graph, param_ids, param_names) = build_graph_for_entries(&entries);
    let mapper =
        WeightMapper::from_param_names_and_ids(&param_names, &param_ids).unwrap();

    let metas: Vec<TensorMeta> = entries
        .iter()
        .map(|(n, s, _)| TensorMeta {
            name: n.to_string(),
            shape: s.clone(),
            dtype: DType::BF16,
        })
        .collect();
    let plan_input = TierPlanInput {
        tensors: metas,
        free_vram_bytes: 16 * 1024 * 1024 * 1024,
        free_ram_bytes: 32 * 1024 * 1024 * 1024,
        model_total_bytes: 0,
        total_ram_bytes: 32 * 1024 * 1024 * 1024,
        kernel_dtype: DType::F32,
    };
    let p = plan(&plan_input);
    assert_eq!(p.get("w_proj.weight"), Some(Tier::Vram));
    assert_eq!(p.get("w_other"), Some(Tier::Ram));

    let (store, _) = mapper
        .load_into_with_residency_plan(&mut graph, &reader, &p, &param_ids, &param_names)
        .expect("load_into_with_residency_plan");

    // Verify variant types.
    let proj = store.get_by_name("w_proj.weight").unwrap();
    let other = store.get_by_name("w_other").unwrap();
    assert!(matches!(proj, SharedParam::Cuda { .. }),
        "w_proj.weight must be SharedParam::Cuda, got {:?}", proj);
    // Default mapper writes F32 to Cpu slots; extract_from_graph
    // hoists Cpu → CpuShared → SharedParam::F32.
    assert!(matches!(other, SharedParam::F32 { .. }),
        "w_other must be SharedParam::F32, got {:?}", other);

    // Numerical correctness: download the GPU tensor and compare
    // against the source BF16-decoded F32 reference.
    let mut gpu_tensor = proj.to_tensor();
    assert!(matches!(gpu_tensor.storage(), TensorStorage::Cuda(_)));
    gpu_tensor.ensure_cpu().expect("D→H download");
    let gpu_values = gpu_tensor.copy_to_cpu_vec();

    // Reference: source values quantised through BF16 → F32
    // (lossless because we encoded them as BF16 originally).
    let src_values = &entries[0].2;
    let expected: Vec<f32> = src_values
        .iter()
        .map(|&v| {
            let bits = f32_to_bf16_bits(v);
            f32::from_bits((bits as u32) << 16)
        })
        .collect();
    assert_eq!(gpu_values.len(), expected.len());
    let mut max_abs_diff = 0.0_f32;
    for (g, e) in gpu_values.iter().zip(expected.iter()) {
        let d = (g - e).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 1e-6,
        "GPU upcast drifted {} from BF16 reference (expected bit-exact)",
        max_abs_diff
    );
}

// ---------------------------------------------------------------------
// Test 3 — Vram fast-path counter: BF16 + no transforms = no F32 transient
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn vram_bf16_no_transforms_uses_fast_path_only() {
    let _guard = COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    // Two `_proj.weight` BF16 tensors with no transforms in the
    // mapper. Plan both to Vram. After load:
    //   - VRAM_FAST_PATH_COUNT increments by exactly 2.
    //   - VRAM_SLOW_PATH_COUNT does NOT move.
    // This proves no `Vec<f32>` was materialised for any Vram
    // upload — the structural guarantee of the M6 replan.
    let entries = vec![
        (
            "alpha_proj.weight",
            vec![4, 4],
            (0..16).map(|i| (i as f32) * 0.05).collect::<Vec<_>>(),
        ),
        (
            "beta_proj.weight",
            vec![3, 5],
            (0..15).map(|i| (i as f32) * 0.07 - 0.3).collect::<Vec<_>>(),
        ),
    ];
    let buf = build_multi_bf16_safetensors(&entries);
    let reader = SafetensorsReader::from_bytes(buf).expect("reader");

    let (mut graph, param_ids, param_names) = build_graph_for_entries(&entries);
    let mapper =
        WeightMapper::from_param_names_and_ids(&param_names, &param_ids).unwrap();

    let metas: Vec<TensorMeta> = entries
        .iter()
        .map(|(n, s, _)| TensorMeta {
            name: n.to_string(),
            shape: s.clone(),
            dtype: DType::BF16,
        })
        .collect();
    let p = plan(&TierPlanInput {
        tensors: metas,
        free_vram_bytes: 16 * 1024 * 1024 * 1024,
        free_ram_bytes: 32 * 1024 * 1024 * 1024,
        model_total_bytes: 0,
        total_ram_bytes: 32 * 1024 * 1024 * 1024,
        kernel_dtype: DType::F32,
    });
    assert_eq!(p.count(Tier::Vram), 2);

    let fast_before = vram_fast_path_count();
    let slow_before = vram_slow_path_count();

    let (store, _) = mapper
        .load_into_with_residency_plan(&mut graph, &reader, &p, &param_ids, &param_names)
        .expect("load_into_with_residency_plan");

    let fast_after = vram_fast_path_count();
    let slow_after = vram_slow_path_count();

    assert_eq!(
        fast_after - fast_before,
        2,
        "expected fast-path count to increment by 2 (no-transform BF16 entries), got {}",
        fast_after - fast_before
    );
    assert_eq!(
        slow_after - slow_before,
        0,
        "slow-path count must not move on a no-transforms BF16 plan; got {} \
         (a non-zero value means an F32 transient was allocated for at least one upload)",
        slow_after - slow_before
    );

    // Both store entries must be Cuda variants.
    assert!(matches!(
        store.get_by_name("alpha_proj.weight").unwrap(),
        SharedParam::Cuda { .. }
    ));
    assert!(matches!(
        store.get_by_name("beta_proj.weight").unwrap(),
        SharedParam::Cuda { .. }
    ));
}
