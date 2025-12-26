use atenia_engine::v13::placement_types::*;
use atenia_engine::v13::tensor_placement::TensorPlacementEngine;
use atenia_engine::v13::types::*;

use std::collections::HashMap;

fn mock_hw_snapshot(
    has_gpu: bool,
    vram_pressure: Option<f32>,
    ram_pressure: Option<f32>,
) -> GlobalHardwareSnapshot {
    GlobalHardwareSnapshot {
        timestamp_epoch_ms: 0,
        cpu: CpuCaps {
            physical_cores: None,
            logical_cores: Some(8),
            simd: vec![],
        },
        gpus: if has_gpu {
            vec![GpuCaps {
                id: "gpu0".to_string(),
                backend: BackendKind::Unknown,
                vram_total_bytes: None,
                vram_free_bytes: None,
                bandwidth_gbps_est: None,
                compute_score_est: None,
            }]
        } else {
            vec![]
        },
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
        reliability_by_device: HashMap::new(),
        pressure: PressureSnapshot {
            vram_pressure,
            ram_pressure,
        },
    }
}

#[test]
fn placement_cpu_when_no_gpu() {
    let hw = mock_hw_snapshot(false, None, None);
    let tensor = TensorProfile {
        num_elements: 1024,
        element_size_bytes: 4,
        estimated_compute_cost: None,
    };

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Cpu);
}

#[test]
fn placement_vram_when_low_pressure() {
    let hw = mock_hw_snapshot(true, Some(0.3), Some(0.1));
    let tensor = TensorProfile {
        num_elements: 1024,
        element_size_bytes: 4,
        estimated_compute_cost: None,
    };

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Vram);
}

#[test]
fn placement_ram_when_vram_high() {
    let hw = mock_hw_snapshot(true, Some(0.95), Some(0.2));
    let tensor = TensorProfile {
        num_elements: 1024,
        element_size_bytes: 4,
        estimated_compute_cost: None,
    };

    let d = TensorPlacementEngine::decide(&tensor, &hw);
    assert_eq!(d.target, PlacementTarget::Ram);
}
