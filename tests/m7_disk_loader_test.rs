//! M7.0 — bit-exact integration test for the `Tier::Disk` arm of
//! `WeightMapper::load_into_with_residency_plan`.
//!
//! This test is the runtime falsifier for the `INVESTIGATION_M7`
//! report's "the Disk loader arm is wired but never executed"
//! finding. It builds a synthetic two-tensor safetensors file,
//! constructs a `TierPlan` that forces one tensor to
//! [`Tier::Disk`], runs the loader, and verifies:
//!
//! 1. The Disk arm actually executed (`disk_slow_path_count`
//!    increments by exactly 1 per Disk-tier entry).
//! 2. The store contains a `SharedParam::Disk` entry for the
//!    targeted name and a `SharedParam::F32` entry for the
//!    other.
//! 3. Materialising the Disk tensor (`to_tensor()` →
//!    `copy_to_cpu_vec()`) produces values bit-exactly equal
//!    to the source after BF16 round-trip.
//!
//! No CUDA needed. Disk I/O lands in
//! `disk_tier::default_cache_dir()` (the M4.7 cache).

use std::collections::HashMap;
use std::sync::Mutex;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::weight_store::SharedParam;
use atenia_engine::gpu::tier_plan::{Tier, TierPlan, TensorMeta};
use atenia_engine::tensor::tensor::{f32_to_bf16_bits, bf16_bits_to_f32, Tensor};
use atenia_engine::tensor::DType;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::{
    disk_fast_path_count, disk_slow_path_count, WeightMapper,
};
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

/// Same serialisation pattern used by `tests/m6_replan_loader_test.rs`.
/// The byte buffers must outlive the views map because `TensorView`
/// borrows them.
fn build_multi_bf16_safetensors(
    entries: &[(&str, Vec<usize>, Vec<f32>)],
) -> Vec<u8> {
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

fn build_graph_for_entries(
    entries: &[(&str, Vec<usize>, Vec<f32>)],
) -> (atenia_engine::amg::graph::Graph, Vec<usize>, Vec<String>) {
    let mut gb = GraphBuilder::new();
    let mut param_ids = Vec::with_capacity(entries.len());
    let param_names: Vec<String> =
        entries.iter().map(|(n, _, _)| n.to_string()).collect();
    for (_, shape, _) in entries {
        let numel: usize = shape.iter().product();
        let pid = gb.parameter(Tensor::new_cpu(shape.clone(), vec![0.0_f32; numel]));
        param_ids.push(pid);
        let _ = gb.output(pid);
    }
    (gb.build(), param_ids, param_names)
}

/// Construct a `TierPlan` by hand — the production planner would
/// only emit `Tier::Disk` when free RAM was tight. For the bit-
/// exact test we want to deterministically force one tensor to
/// Disk regardless of host memory state.
fn handcrafted_plan(disk_name: &str, ram_name: &str) -> TierPlan {
    // The public `plan(input)` API doesn't accept manual overrides,
    // but we can drive the loader with a `TierPlan` produced by
    // `plan(input)` whose inputs we calibrate so the bin packer
    // emits exactly Disk + Ram. The simplest way: zero VRAM, RAM
    // budget that fits the second tensor only.
    //
    // Tensor sizes (BF16): both 4×4 = 32 bytes.
    // RAM headroom in `plan` is hardcoded at 8 GiB. To force one
    // tensor to Disk we'd need free RAM ≤ 8 GiB + 32 bytes — this
    // is not representable as a synthetic test on the live machine
    // probe.
    //
    // Instead, build the plan via `plan` with a normal-looking
    // input so all entries land on Ram, then post-process: rebuild
    // the `assignments` and `by_name` HashMap with the Disk
    // override. This requires accessing private fields, which we
    // can't do from an integration test.
    //
    // Workaround: drive `plan` with a free_ram_bytes value that
    // makes RAM headroom barely cover the smallest tensor. With
    // both tensors at 32 bytes BF16 + 8 GiB hardcoded floor: set
    // `free_ram_bytes = 8 GiB + 32` so the planner emits 1 Ram + 1
    // Disk in input order.
    use atenia_engine::gpu::tier_plan::{plan, TierPlanInput};

    let metas = vec![
        // First entry → Ram (gets the only RAM slot).
        TensorMeta {
            name: ram_name.to_string(),
            shape: vec![4, 4],
            dtype: DType::BF16,
        },
        // Second entry → overflows to Disk because RAM is exhausted.
        TensorMeta {
            name: disk_name.to_string(),
            shape: vec![4, 4],
            dtype: DType::BF16,
        },
    ];
    // Both tensors are 32 bytes BF16; total 64 bytes. Far below the
    // M7.2 adaptive threshold (0.7 × free_ram), so the headroom
    // stays at the 8 GiB base and the calibration above remains
    // valid: 1 Ram + 1 Disk in input order.
    let model_total = 64_u64;
    let p = plan(&TierPlanInput {
        tensors: metas,
        free_vram_bytes: 0, // Force everything off VRAM.
        // 8 GiB headroom + room for exactly one 32-byte tensor.
        free_ram_bytes: 8 * 1024 * 1024 * 1024 + 32,
        model_total_bytes: model_total,
        total_ram_bytes: 16 * 1024 * 1024 * 1024,
    });
    assert_eq!(p.get(ram_name), Some(Tier::Ram));
    assert_eq!(p.get(disk_name), Some(Tier::Disk));
    p
}

static DISK_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn disk_tier_loader_arm_executes_and_round_trips_bit_exact() {
    // Serialise with sibling tests in this file (none today, but
    // future M7 tests will share the global atomic counters).
    let _guard = DISK_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // Two synthetic BF16 tensors. `disk_target` is forced to
    // `Tier::Disk`; `ram_resident` stays on RAM. Values are
    // deterministic so the bit-exact comparison is reproducible.
    let disk_values: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let ram_values: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.3).collect();
    let entries = vec![
        ("ram_resident", vec![4, 4], ram_values.clone()),
        ("disk_target", vec![4, 4], disk_values.clone()),
    ];

    let buf = build_multi_bf16_safetensors(&entries);
    let reader = SafetensorsReader::from_bytes(buf).expect("reader");

    let (mut graph, param_ids, param_names) = build_graph_for_entries(&entries);
    let mapper =
        WeightMapper::from_param_names_and_ids(&param_names, &param_ids).unwrap();

    let plan = handcrafted_plan("disk_target", "ram_resident");

    // Snapshot the Disk slow-path counter before running the
    // loader. The increment delta is the runtime proof that the
    // Disk arm actually executed — which is the falsifier the M7
    // investigation flagged.
    let slow_before = disk_slow_path_count();

    let (store, _report) = mapper
        .load_into_with_residency_plan(
            &mut graph,
            &reader,
            &plan,
            &param_ids,
            &param_names,
        )
        .expect("load_into_with_residency_plan");

    let slow_after = disk_slow_path_count();

    // ----- (1) Counter must show the Disk arm executed exactly once.
    assert_eq!(
        slow_after - slow_before,
        1,
        "DISK_SLOW_PATH_COUNT did not increment by exactly 1; the \
         Disk arm of load_into_with_residency_plan was not exercised \
         (delta = {}). This confirms the M7 investigation finding \
         that the Disk loader path was compile-checked but never \
         executed at runtime — the test is the runtime falsifier.",
        slow_after - slow_before
    );

    // ----- (2) Variant types in the store.
    let disk_entry = store.get_by_name("disk_target").unwrap();
    let ram_entry = store.get_by_name("ram_resident").unwrap();
    assert!(
        matches!(disk_entry, SharedParam::Disk { .. }),
        "disk_target must be SharedParam::Disk, got {:?}",
        disk_entry
    );
    assert!(
        matches!(ram_entry, SharedParam::F32 { .. }),
        "ram_resident must be SharedParam::F32 (default mapper, Cpu \
         graph slot, hoisted by extract_from_graph), got {:?}",
        ram_entry
    );

    // ----- (3) Bit-exact round-trip on the Disk-resident tensor.
    let disk_tensor = disk_entry.to_tensor();
    let restored: Vec<f32> = disk_tensor.copy_to_cpu_vec();

    // Reference: source values quantised through BF16 → F32. The
    // BF16 round-trip is bit-exact by construction (high 16 bits
    // of F32, mantissa LSBs zeroed).
    let expected: Vec<f32> = disk_values
        .iter()
        .map(|&v| {
            let bits = f32_to_bf16_bits(v);
            bf16_bits_to_f32(bits)
        })
        .collect();

    assert_eq!(restored.len(), expected.len());
    for (i, (r, e)) in restored.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            e.to_bits(),
            "bit-exact mismatch at index {} (disk={:e}, expected={:e})",
            i,
            r,
            e
        );
    }
}

/// **M7.1** — Disk fast-path validation. Same setup as the M7.0
/// test above, but with the mapper configured for BF16 storage
/// (`store_params_as_bf16(true)`). Under that mode + BF16 source
/// + no transforms, the loader must take the fast path:
///   - `DISK_FAST_PATH_COUNT` increments by 1.
///   - `DISK_SLOW_PATH_COUNT` does NOT move.
///   - The on-disk byte layout is bit-exact equivalent to what
///     the slow path would produce (verified indirectly by
///     reading back and comparing values).
#[test]
fn disk_fast_path_with_bf16_store_no_transforms() {
    let _guard = DISK_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let disk_values: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let ram_values: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.3).collect();
    let entries = vec![
        ("ram_resident_fp", vec![4, 4], ram_values.clone()),
        ("disk_target_fp", vec![4, 4], disk_values.clone()),
    ];

    let buf = build_multi_bf16_safetensors(&entries);
    let reader = SafetensorsReader::from_bytes(buf).expect("reader");

    let (mut graph, param_ids, param_names) = build_graph_for_entries(&entries);
    let mut mapper =
        WeightMapper::from_param_names_and_ids(&param_names, &param_ids).unwrap();
    // **Critical for the fast path**: enable BF16 storage on the
    // mapper. The fast path only fires under this config because
    // F32-on-disk requires the F32 transient anyway.
    mapper.set_store_params_as_bf16(true);

    let plan = handcrafted_plan("disk_target_fp", "ram_resident_fp");

    let fast_before = disk_fast_path_count();
    let slow_before = disk_slow_path_count();

    let (store, _report) = mapper
        .load_into_with_residency_plan(
            &mut graph,
            &reader,
            &plan,
            &param_ids,
            &param_names,
        )
        .expect("load_into_with_residency_plan");

    let fast_after = disk_fast_path_count();
    let slow_after = disk_slow_path_count();

    assert_eq!(
        fast_after - fast_before,
        1,
        "DISK_FAST_PATH_COUNT must increment by 1 when source is BF16 \
         + no transforms + store_params_as_bf16=true; got delta {}",
        fast_after - fast_before
    );
    assert_eq!(
        slow_after - slow_before,
        0,
        "DISK_SLOW_PATH_COUNT must NOT move on the fast-path scenario; \
         got delta {} (a non-zero value means an F32 transient was \
         materialised, defeating the M7.1 peak-RAM contract)",
        slow_after - slow_before
    );

    // Bit-exact round-trip via to_tensor + copy_to_cpu_vec, same
    // contract as the M7.0 slow-path test.
    let disk_entry = store.get_by_name("disk_target_fp").unwrap();
    assert!(matches!(disk_entry, SharedParam::Disk { .. }));
    let disk_tensor = disk_entry.to_tensor();
    let restored: Vec<f32> = disk_tensor.copy_to_cpu_vec();

    let expected: Vec<f32> = disk_values
        .iter()
        .map(|&v| {
            let bits = f32_to_bf16_bits(v);
            bf16_bits_to_f32(bits)
        })
        .collect();

    assert_eq!(restored.len(), expected.len());
    for (i, (r, e)) in restored.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            e.to_bits(),
            "fast-path bit-exact mismatch at index {} (disk={:e}, expected={:e})",
            i,
            r,
            e
        );
    }
}
