use atenia_engine::v13::placement_types::*;
use atenia_engine::v13::tensor_placement::TensorPlacementEngine;
use atenia_engine::v13::types::*;

use std::collections::HashMap;

fn make_gpu(id: &str, compute_score_est: Option<f32>) -> GpuCaps {
    GpuCaps {
        id: id.to_string(),
        backend: BackendKind::Unknown,
        vram_total_bytes: None,
        vram_free_bytes: None,
        bandwidth_gbps_est: None,
        compute_score_est,
    }
}

fn mock_hw_snapshot_single_gpu(
    vram_pressure: Option<f32>,
    ram_pressure: Option<f32>,
) -> GlobalHardwareSnapshot {
    let gpus = vec![make_gpu("gpu0", Some(10.0))];
    let reliability_by_device: HashMap<String, ReliabilityStats> = HashMap::new();

    GlobalHardwareSnapshot {
        timestamp_epoch_ms: 0,
        cpu: CpuCaps {
            physical_cores: None,
            logical_cores: Some(8),
            simd: vec![],
        },
        gpus,
        ram: RamCaps {
            total_bytes: None,
            free_bytes: None,
        },
        ssd: SsdCaps {
            cache_dir: ".cache".to_string(),
            read_mb_s_est: None,
            write_mb_s_est: None,
            latency_ms_est: None,
        },
        reliability_by_device,
        pressure: PressureSnapshot {
            vram_pressure,
            ram_pressure,
        },
    }
}

fn make_tensor(num_elements: u64, element_size_bytes: u32, estimated_compute_cost: Option<f32>) -> TensorProfile {
    TensorProfile {
        num_elements,
        element_size_bytes,
        estimated_compute_cost,
    }
}

fn huge_tensor_elements_for_4b() -> u64 {
    (512 * 1024 * 1024 / 4) as u64 + 1
}

#[test]
fn ssd_when_vram_and_ram_critical_and_tensor_huge() {
    let hw = mock_hw_snapshot_single_gpu(Some(0.98), Some(0.97));
    let tensor = make_tensor(huge_tensor_elements_for_4b(), 4, None);

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Ssd);
    assert!(d.device_id.is_none());
    assert!(!d.reason.is_empty());
}

#[test]
fn cpu_when_critical_but_tensor_small() {
    let hw = mock_hw_snapshot_single_gpu(Some(0.98), Some(0.97));
    let tensor = make_tensor(1024, 4, None);

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Cpu);
    assert!(d.device_id.is_none());
    assert!(!d.reason.is_empty());
}

#[test]
fn ram_when_vram_critical_but_ram_safe() {
    let hw = mock_hw_snapshot_single_gpu(Some(0.98), Some(0.20));
    let tensor = make_tensor(1024, 4, None);

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Ram);
    assert!(d.device_id.is_none());
    assert!(!d.reason.is_empty());
}

#[test]
fn gpu_still_wins_when_compute_heavy_and_vram_safe() {
    let hw = mock_hw_snapshot_single_gpu(Some(0.40), Some(0.90));
    let tensor = make_tensor(1024, 4, Some(200.0));

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Vram);
    assert_eq!(d.device_id.as_deref(), Some("gpu0"));
    assert!(!d.reason.is_empty());
}
