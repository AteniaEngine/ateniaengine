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

fn make_stats(ok_count: u64, fail_count: u64, has_recent_error: bool) -> ReliabilityStats {
    ReliabilityStats {
        ok_count,
        fail_count,
        last_error: if has_recent_error {
            Some("last error".to_string())
        } else {
            None
        },
        last_error_epoch_ms: if has_recent_error { Some(1234) } else { None },
    }
}

fn mock_hw_snapshot_multi_gpu(
    gpus: Vec<GpuCaps>,
    vram_pressure: Option<f32>,
    ram_pressure: Option<f32>,
    reliability: Vec<(String, ReliabilityStats)>,
) -> GlobalHardwareSnapshot {
    let mut reliability_by_device = HashMap::new();
    for (id, stats) in reliability {
        reliability_by_device.insert(id, stats);
    }

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

#[test]
fn best_gpu_by_reliability() {
    let gpus = vec![
        make_gpu("gpu0", Some(10.0)),
        make_gpu("gpu1", Some(5.0)),
    ];

    let reliability = vec![
        ("gpu0".to_string(), make_stats(2, 8, false)),
        ("gpu1".to_string(), make_stats(9, 1, false)),
    ];

    let hw = mock_hw_snapshot_multi_gpu(gpus, Some(0.2), Some(0.1), reliability);
    let tensor = make_tensor(1024, 4, None);

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Vram);
    assert_eq!(d.device_id.as_deref(), Some("gpu1"));
}

#[test]
fn tie_breaker_by_compute_score() {
    let gpus = vec![
        make_gpu("gpu0", Some(5.0)),
        make_gpu("gpu1", Some(7.0)),
    ];

    let reliability = vec![
        ("gpu0".to_string(), make_stats(10, 0, false)),
        ("gpu1".to_string(), make_stats(10, 0, false)),
    ];

    let hw = mock_hw_snapshot_multi_gpu(gpus, Some(0.2), Some(0.1), reliability);
    let tensor = make_tensor(1024, 4, None);

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Vram);
    assert_eq!(d.device_id.as_deref(), Some("gpu1"));
}

#[test]
fn huge_tensor_prefers_ram_when_pressure_allows() {
    // > 512MB assuming 4-byte elements
    let huge_elements = (512 * 1024 * 1024 / 4) as u64 + 1;
    let tensor = make_tensor(huge_elements, 4, None);

    let gpus = vec![make_gpu("gpu0", Some(10.0))];
    let reliability = vec![("gpu0".to_string(), make_stats(10, 0, false))];

    let hw = mock_hw_snapshot_multi_gpu(gpus, Some(0.85), Some(0.2), reliability);

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Ram);
    assert_eq!(d.device_id, None);
}

#[test]
fn compute_heavy_prefers_gpu_when_safe() {
    let tensor = make_tensor(1024, 4, Some(200.0));

    let gpus = vec![make_gpu("gpu0", Some(10.0))];
    let reliability = vec![("gpu0".to_string(), make_stats(10, 0, false))];

    let hw = mock_hw_snapshot_multi_gpu(gpus, Some(0.4), Some(0.2), reliability);

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Vram);
    assert_eq!(d.device_id.as_deref(), Some("gpu0"));
}

#[test]
fn high_pressure_forces_cpu() {
    let tensor = make_tensor(1024, 4, Some(200.0));

    let gpus = vec![make_gpu("gpu0", Some(10.0))];
    let reliability = vec![("gpu0".to_string(), make_stats(10, 0, false))];

    let hw = mock_hw_snapshot_multi_gpu(gpus, Some(0.98), Some(0.97), reliability);

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Cpu);
    assert_eq!(d.device_id, None);
}
