//! M8.4 — integration tests for the BF16 end-to-end wire-up.
//!
//! Covers three contracts:
//!
//! 1. **Loader gate under flag**: with `ATENIA_M8_BF16_KERNEL=1`,
//!    the loader's `Tier::Vram + dtype=BF16 + no transforms` arm
//!    routes through `bf16_to_vram_no_upcast_from_raw_bytes` (the
//!    M8.1 primitive) and the resulting `SharedParam::Cuda` carries
//!    a BF16-typed `TensorGPU`. Counter `vram_bf16_fast_path_count`
//!    advances by 1 per Vram-tier _proj.weight; `vram_fast_path_count`
//!    (the M6 F32 path) stays flat.
//!
//! 2. **Default loader stays F32 (regression-zero)**: with the env
//!    var unset, the loader takes the M6 path bit-identically — the
//!    `SharedParam::Cuda` carries an F32 `TensorGPU`,
//!    `vram_fast_path_count` advances, `vram_bf16_fast_path_count`
//!    stays flat. This is the contract that lets us ship M8.4 to
//!    operators without flipping the default.
//!
//! 3. **Dispatcher routes BF16-resident triples to cublasGemmEx**:
//!    `try_gpu_matmul` recognises `(Cpu F32, Cuda BF16, Cpu F32)` and
//!    routes to `cuda_matmul_bf16_inplace` (M8.2). Output drift vs a
//!    CPU F32 reference stays under the M8.2 single-op envelope
//!    (0.5; ADR-004 end-to-end is M8.5's gate). Counter
//!    `vram_bf16_matmul_count` advances by 1.
//!
//! All three tests require CUDA and are gated behind a process-wide
//! `ENV_TEST_LOCK` because they manipulate `ATENIA_M8_BF16_KERNEL`,
//! which is process-global. Without serialisation, two parallel
//! tests would observe each other's flag flip.

use std::collections::HashMap;
use std::sync::Mutex;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::weight_store::SharedParam;
use atenia_engine::cuda::cuda_available;
use atenia_engine::cuda::matmul::vram_bf16_matmul_count;
use atenia_engine::gpu::dispatch::hooks::try_gpu_matmul;
use atenia_engine::gpu::tier_plan::{plan, TensorMeta, Tier, TierPlanInput};
use atenia_engine::tensor::tensor::{f32_to_bf16_bits, Tensor};
use atenia_engine::tensor::{DType, TensorStorage};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::{
    vram_bf16_fast_path_count, vram_bf16_slow_path_count, vram_fast_path_count,
    vram_slow_path_count, LoadTransform, WeightMapper,
};
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

/// Process-wide serialisation for tests that touch
/// `ATENIA_M8_BF16_KERNEL` and global counters. The flag is a
/// module-level read at every loader invocation; without this
/// lock, parallel tests would race on whether the flag is set.
static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard that sets `ATENIA_M8_BF16_KERNEL` on construction
/// and removes it on `Drop`. Even if the test panics, `Drop` runs
/// during stack unwind so the env var is cleaned up — no
/// poisoning of subsequent tests via leaked global state.
struct M8FlagGuard;

impl M8FlagGuard {
    fn set() -> Self {
        // SAFETY: `ENV_TEST_LOCK` holder must be acquired by the
        // caller. set_var is safe under that mutex because every
        // test that reads the var also holds the lock.
        unsafe {
            std::env::set_var("ATENIA_M8_BF16_KERNEL", "1");
        }
        Self
    }
}

impl Drop for M8FlagGuard {
    fn drop(&mut self) {
        unsafe {
            std::env::remove_var("ATENIA_M8_BF16_KERNEL");
        }
    }
}

// ---------------------------------------------------------------------
// Synthetic safetensors helpers — copied from m6_replan_loader_test
// so this file is self-contained.
// ---------------------------------------------------------------------

fn build_multi_bf16_safetensors(entries: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
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
    let param_names: Vec<String> = entries.iter().map(|(n, _, _)| n.to_string()).collect();
    for (_, shape, _) in entries {
        let numel: usize = shape.iter().product();
        let pid = gb.parameter(Tensor::new_cpu(shape.clone(), vec![0.0_f32; numel]));
        param_ids.push(pid);
        let _ = gb.output(pid);
    }
    let graph = gb.build();
    (graph, param_ids, param_names)
}

fn make_metas(entries: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<TensorMeta> {
    entries
        .iter()
        .map(|(n, s, _)| TensorMeta {
            name: n.to_string(),
            shape: s.clone(),
            dtype: DType::BF16,
        })
        .collect()
}

// ---------------------------------------------------------------------
// Test 1 — loader under flag routes to BF16-resident path.
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn m8_4_loader_routes_proj_weight_to_bf16_vram_under_flag() {
    let _guard = ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    // Set the M8 flag for this test only. RAII guard removes
    // it at the end of the test (panic-safe).
    let _flag = M8FlagGuard::set();

    let entries = vec![(
        "w_proj.weight",
        vec![8, 4],
        (0..32).map(|i| (i as f32) * 0.05 - 0.5).collect::<Vec<_>>(),
    )];
    let buf = build_multi_bf16_safetensors(&entries);
    let reader = SafetensorsReader::from_bytes(buf).expect("reader");
    let (mut graph, param_ids, param_names) = build_graph_for_entries(&entries);
    let mapper =
        WeightMapper::from_param_names_and_ids(&param_names, &param_ids).unwrap();

    // Plan with abundant VRAM so the proj weight lands on Vram.
    // `kernel_dtype: BF16` is the M8.3 input that lets the planner
    // count weights at numel × 2.
    let plan_input = TierPlanInput {
        tensors: make_metas(&entries),
        free_vram_bytes: 16 * 1024 * 1024 * 1024,
        free_ram_bytes: 32 * 1024 * 1024 * 1024,
        model_total_bytes: 0,
        total_ram_bytes: 32 * 1024 * 1024 * 1024,
        kernel_dtype: DType::BF16,
    };
    let p = plan(&plan_input);
    assert_eq!(p.get("w_proj.weight"), Some(Tier::Vram));

    // Snapshot counters.
    let m8_before = vram_bf16_fast_path_count();
    let m6_before = vram_fast_path_count();
    let slow_before = vram_slow_path_count();

    let (store, _) = mapper
        .load_into_with_residency_plan(&mut graph, &reader, &p, &param_ids, &param_names)
        .expect("load_into_with_residency_plan");

    let m8_after = vram_bf16_fast_path_count();
    let m6_after = vram_fast_path_count();
    let slow_after = vram_slow_path_count();

    // BF16 fast-path must have advanced by exactly 1.
    assert_eq!(
        m8_after - m8_before,
        1,
        "expected vram_bf16_fast_path_count += 1 under flag, got delta {}",
        m8_after - m8_before
    );
    // M6 F32 fast-path must NOT have advanced.
    assert_eq!(
        m6_after, m6_before,
        "vram_fast_path_count must not advance under M8.4 flag (got delta {})",
        m6_after - m6_before
    );
    // Slow path must NOT have advanced (BF16 + no-transforms is fast).
    assert_eq!(slow_after, slow_before);

    // The store entry is `SharedParam::Cuda` with a **BF16**-typed
    // TensorGPU. This is the structural assertion that distinguishes
    // M8.4 from M6 — same enum variant, different dtype on the
    // inner TensorGPU.
    let proj = store.get_by_name("w_proj.weight").unwrap();
    let gpu = match proj {
        SharedParam::Cuda { gpu, .. } => gpu,
        other => panic!("expected SharedParam::Cuda, got {:?}", other),
    };
    assert_eq!(
        gpu.dtype(),
        DType::BF16,
        "M8.4 path must produce BF16-resident TensorGPU"
    );
    assert_eq!(
        gpu.size_bytes(),
        32 * 2,
        "BF16 buffer must be numel × 2 = 64 bytes; got {}",
        gpu.size_bytes()
    );

    // Round-trip the BF16 device buffer back to host and verify
    // every u16 bit pattern matches the source.
    let downloaded = gpu
        .to_cpu_bf16_bits()
        .expect("to_cpu_bf16_bits on BF16-resident gpu");
    let src_bits: Vec<u16> = entries[0]
        .2
        .iter()
        .map(|&v| f32_to_bf16_bits(v))
        .collect();
    assert_eq!(downloaded, src_bits);
}

// ---------------------------------------------------------------------
// Test 2 — default loader stays on F32 path (regression-zero).
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn m8_4_default_loader_keeps_f32_path_with_no_flag() {
    let _guard = ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    // Defensive: if the env var is somehow set from outside the
    // test run, clear it for this test. The RAII guard would
    // restore on drop — we want to OBSERVE the unset state.
    unsafe {
        std::env::remove_var("ATENIA_M8_BF16_KERNEL");
    }

    let entries = vec![(
        "w_proj.weight",
        vec![8, 4],
        (0..32).map(|i| (i as f32) * 0.05 - 0.5).collect::<Vec<_>>(),
    )];
    let buf = build_multi_bf16_safetensors(&entries);
    let reader = SafetensorsReader::from_bytes(buf).expect("reader");
    let (mut graph, param_ids, param_names) = build_graph_for_entries(&entries);
    let mapper =
        WeightMapper::from_param_names_and_ids(&param_names, &param_ids).unwrap();

    let plan_input = TierPlanInput {
        tensors: make_metas(&entries),
        free_vram_bytes: 16 * 1024 * 1024 * 1024,
        free_ram_bytes: 32 * 1024 * 1024 * 1024,
        model_total_bytes: 0,
        total_ram_bytes: 32 * 1024 * 1024 * 1024,
        // Plan with F32 kernel_dtype to mirror the default
        // production path; same as M6 / M7.
        kernel_dtype: DType::F32,
    };
    let p = plan(&plan_input);
    assert_eq!(p.get("w_proj.weight"), Some(Tier::Vram));

    let m8_before = vram_bf16_fast_path_count();
    let m6_before = vram_fast_path_count();

    let (store, _) = mapper
        .load_into_with_residency_plan(&mut graph, &reader, &p, &param_ids, &param_names)
        .expect("load_into_with_residency_plan");

    let m8_after = vram_bf16_fast_path_count();
    let m6_after = vram_fast_path_count();

    // M6 F32 fast-path must have advanced.
    assert_eq!(
        m6_after - m6_before,
        1,
        "expected vram_fast_path_count += 1 under default path, got delta {}",
        m6_after - m6_before
    );
    // BF16 fast-path must NOT have advanced.
    assert_eq!(
        m8_after, m8_before,
        "vram_bf16_fast_path_count must stay flat without the flag (got delta {})",
        m8_after - m8_before
    );

    let proj = store.get_by_name("w_proj.weight").unwrap();
    let gpu = match proj {
        SharedParam::Cuda { gpu, .. } => gpu,
        other => panic!("expected SharedParam::Cuda, got {:?}", other),
    };
    assert_eq!(
        gpu.dtype(),
        DType::F32,
        "default path must produce F32-resident TensorGPU (M6 contract)"
    );
}

// ---------------------------------------------------------------------
// Test 3 — dispatcher routes BF16-resident triples to cuda_matmul_bf16.
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn m8_4_dispatcher_routes_bf16_resident_to_cuda_matmul_bf16() {
    let _guard = ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }

    // Mid-size shape: [m=1, k=128] × [k=128, n=64].
    let m = 1_usize;
    let k = 128_usize;
    let n = 64_usize;

    // Synth deterministic data, magnitude ~ 0.3.
    let a_host: Vec<f32> = (0..(m * k))
        .map(|i| ((i as f32) * 0.013 + 0.7).sin() * 0.3)
        .collect();
    let b_host_f32: Vec<f32> = (0..(k * n))
        .map(|i| ((i as f32) * 0.007 + 0.4).cos() * 0.3)
        .collect();

    // BF16-encode `b` and upload via the M8.1 primitive.
    let b_bf16: Vec<u16> =
        b_host_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let gpu = atenia_engine::cuda::bf16_to_f32::bf16_to_vram_no_upcast(&b_bf16, &[k, n])
        .expect("bf16_to_vram_no_upcast on a CUDA host");
    let b_tensor = Tensor::from_cuda_gpu(vec![k, n], gpu);

    // Activation `a` and output `out` stay on host as F32.
    let a_tensor = Tensor::new_cpu(vec![m, k], a_host.clone());
    let mut out_tensor = Tensor::new_cpu(vec![m, n], vec![0.0_f32; m * n]);

    // Dispatch.
    let bf16_before = vram_bf16_matmul_count();
    let ok = try_gpu_matmul(&a_tensor, &b_tensor, &mut out_tensor);
    let bf16_after = vram_bf16_matmul_count();

    assert!(ok, "try_gpu_matmul must accept the BF16-resident triple");
    assert_eq!(
        bf16_after - bf16_before,
        1,
        "vram_bf16_matmul_count expected += 1, got delta {}",
        bf16_after - bf16_before
    );
    assert!(
        matches!(out_tensor.storage(), TensorStorage::Cpu(_)),
        "out must remain on Cpu storage after BF16 dispatch"
    );

    // Reference: F32 CPU matmul against the BF16-decoded weight
    // (so the drift measured is the cumulative BF16 truncation
    // envelope on both operands, not BF16 + F32 mixed).
    let b_ref_f32: Vec<f32> = b_bf16
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let mut out_ref = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for p in 0..k {
                s += a_host[i * k + p] * b_ref_f32[p * n + j];
            }
            out_ref[i * n + j] = s;
        }
    }

    // Drift gate (single-op envelope, same rationale as
    // `cuda_matmul_bf16_inplace_drift_within_gate_on_llama_13b_shapes`).
    let out = out_tensor.as_cpu_slice();
    let mut max_abs_diff = 0.0_f32;
    for (g, r) in out.iter().zip(out_ref.iter()) {
        let d = (g - r).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    eprintln!(
        "[M8.4] dispatcher drift on [m=1, k=128, n=64]: {:.4e}",
        max_abs_diff
    );
    assert!(
        max_abs_diff < 0.5,
        "BF16 dispatcher drift {:.4e} exceeded 0.5 envelope",
        max_abs_diff
    );
}

// ---------------------------------------------------------------------
// Test 4 — loader slow path under flag (LoadTransform::Transpose2D).
//
// **M8.4b** — exercises the gap M8.5 surfaced: every Llama-family
// `_proj.weight` has at least `LoadTransform::Transpose2D` registered,
// so the M8.4-original fast-path arm (`mapping.transforms.is_empty()`)
// never fired in production. The slow path now has a parallel M8
// branch: F32 transforms run, then the result re-encodes to BF16 and
// uploads via `bf16_to_vram_no_upcast`. This test verifies that wire-up
// end-to-end on a synthetic mapper that registers Transpose2D — same
// transform shape every Llama proj weight uses.
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA driver (nvidia-smi)"]
fn m8_4b_loader_routes_proj_weight_with_transforms_to_bf16_vram_under_flag() {
    let _guard = ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let _flag = M8FlagGuard::set();

    // [4, 4] BF16 proj weight; Transpose2D after load gives [4, 4]
    // (same shape, but the transform fires the slow path). The
    // builder's parameter slot is shape-allocated for the **post-
    // transform** shape, so we use [4, 4] uniformly here.
    let entries = vec![(
        "w_proj.weight",
        vec![4, 4],
        (0..16).map(|i| (i as f32) * 0.05 - 0.4).collect::<Vec<_>>(),
    )];
    let buf = build_multi_bf16_safetensors(&entries);
    let reader = SafetensorsReader::from_bytes(buf).expect("reader");
    let (mut graph, param_ids, param_names) = build_graph_for_entries(&entries);

    // Build the mapper and register a Transpose2D transform on
    // the proj weight — this is the canonical Llama _proj.weight
    // recipe per `compute_transforms_for_name`.
    let mut mapper =
        WeightMapper::from_param_names_and_ids(&param_names, &param_ids).unwrap();
    mapper
        .set_transforms("w_proj.weight", vec![LoadTransform::Transpose2D])
        .expect("set Transpose2D transform");

    // Plan with abundant VRAM so the proj weight lands on Vram
    // and `kernel_dtype: BF16` so M8.3 counts at numel × 2.
    let plan_input = TierPlanInput {
        tensors: make_metas(&entries),
        free_vram_bytes: 16 * 1024 * 1024 * 1024,
        free_ram_bytes: 32 * 1024 * 1024 * 1024,
        model_total_bytes: 0,
        total_ram_bytes: 32 * 1024 * 1024 * 1024,
        kernel_dtype: DType::BF16,
    };
    let p = plan(&plan_input);
    assert_eq!(p.get("w_proj.weight"), Some(Tier::Vram));

    // Snapshot all four counters — the transforms-aware test must
    // increment exactly the BF16 SLOW counter and leave the other
    // three flat.
    let m8_fast_before = vram_bf16_fast_path_count();
    let m8_slow_before = vram_bf16_slow_path_count();
    let m6_fast_before = vram_fast_path_count();
    let m6_slow_before = vram_slow_path_count();

    let (store, _) = mapper
        .load_into_with_residency_plan(&mut graph, &reader, &p, &param_ids, &param_names)
        .expect("load_into_with_residency_plan with transforms");

    let m8_fast_after = vram_bf16_fast_path_count();
    let m8_slow_after = vram_bf16_slow_path_count();
    let m6_fast_after = vram_fast_path_count();
    let m6_slow_after = vram_slow_path_count();

    // Counter contract: only `vram_bf16_slow_path_count` advanced.
    assert_eq!(
        m8_slow_after - m8_slow_before,
        1,
        "expected vram_bf16_slow_path_count += 1 under flag with transforms, \
         got delta {}",
        m8_slow_after - m8_slow_before
    );
    assert_eq!(
        m8_fast_after, m8_fast_before,
        "vram_bf16_fast_path_count must NOT advance with transforms registered \
         (got delta {})",
        m8_fast_after - m8_fast_before
    );
    assert_eq!(
        m6_fast_after, m6_fast_before,
        "vram_fast_path_count (M6 F32 fast) must NOT advance under M8 flag"
    );
    assert_eq!(
        m6_slow_after, m6_slow_before,
        "vram_slow_path_count (M6 F32 slow) must NOT advance under M8 flag"
    );

    // The store entry is `SharedParam::Cuda` with a BF16-typed
    // TensorGPU — same contract as the fast-path test, even
    // though the loader took a different code path.
    let proj = store.get_by_name("w_proj.weight").unwrap();
    let gpu = match proj {
        SharedParam::Cuda { gpu, .. } => gpu,
        other => panic!("expected SharedParam::Cuda, got {:?}", other),
    };
    assert_eq!(
        gpu.dtype(),
        DType::BF16,
        "M8.4b slow path must produce BF16-resident TensorGPU"
    );
    assert_eq!(
        gpu.size_bytes(),
        16 * 2,
        "BF16 buffer must be numel × 2 = 32 bytes; got {}",
        gpu.size_bytes()
    );

    // The downloaded bytes should be the BF16-encoded
    // **transposed** F32 — i.e. transposing the raw source values
    // to F32, then BF16-rounding. We verify by comparing element-
    // wise against the manually transposed reference.
    let downloaded = gpu
        .to_cpu_bf16_bits()
        .expect("to_cpu_bf16_bits on BF16-resident gpu");

    // Reference: take the BF16-rounded source values (what's
    // actually serialised on disk by `build_multi_bf16_safetensors`),
    // transpose, and compare bit-for-bit. The loader pipeline is:
    //   disk BF16 → F32 decode (bit-exact) → Transpose2D → BF16
    //   re-encode (bit-exact for BF16-aligned F32) → upload.
    // So the expected output bits equal the disk bits permuted by
    // transpose, with no rounding losses introduced by the slow
    // path itself.
    let src = &entries[0].2;
    let bf16_on_disk: Vec<u16> = src.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let mut expected_bits = vec![0_u16; 16];
    for i in 0..4 {
        for j in 0..4 {
            expected_bits[j * 4 + i] = bf16_on_disk[i * 4 + j];
        }
    }
    assert_eq!(
        downloaded, expected_bits,
        "BF16 slow-path output mismatch — transforms or re-encode regressed"
    );
}
