use atenia_engine::apx5_4::{
    AdaptiveSelector,
    Sample,
    DeviceTarget,
};
use atenia_engine::apx5::apx_5_3_planner::NodeExecInfo;
use atenia_engine::tensor::DType;

fn make_info(shape: Vec<usize>, dtype: DType, device_52: DeviceTarget) -> NodeExecInfo {
    NodeExecInfo {
        node_id: 0,
        op_name: "MatMul".to_string(),
        shape,
        dtype: format!("{:?}", dtype),
        contiguous: true,
        device_52: match device_52 {
            DeviceTarget::CPU => "CPU".to_string(),
            DeviceTarget::GPU => "GPU".to_string(),
        },
        estimated_bytes: 0,
        estimated_flops: 0,
        vram_free: 0,
        kernel_time_avg: 0.0,
        preferred_kernel_size: None,
        tile_override: None,
        scheduling_bias: None,
        qkv_bias: None,
        attention_bias: None,
        exec_priority: None,
        prefetch_hint: None,
    }
}

#[test]
fn adaptive_prefers_cpu_when_cpu_faster() {
    let mut selector = AdaptiveSelector::new();

    selector.register_sample(Sample {
        op_name: "MatMul".into(),
        shape: vec![64, 64],
        dtype: DType::F32,
        device_chosen: DeviceTarget::CPU,
        duration_us: 100,
        vram_before: 0,
        vram_after: 0,
        fallback: false,
    });

    selector.register_sample(Sample {
        op_name: "MatMul".into(),
        shape: vec![64, 64],
        dtype: DType::F32,
        device_chosen: DeviceTarget::GPU,
        duration_us: 350,
        vram_before: 0,
        vram_after: 0,
        fallback: false,
    });

    let info = make_info(vec![64, 64], DType::F32, DeviceTarget::CPU);

    let decision = selector.decide(&info);
    assert!(matches!(decision.prefer_device, Some(DeviceTarget::CPU)));
}

#[test]
fn adaptive_prefers_gpu_when_gpu_faster_and_reliable() {
    let mut selector = AdaptiveSelector::new();

    selector.register_sample(Sample {
        op_name: "MatMul".into(),
        shape: vec![128, 128],
        dtype: DType::F32,
        device_chosen: DeviceTarget::CPU,
        duration_us: 500,
        vram_before: 0,
        vram_after: 0,
        fallback: false,
    });

    selector.register_sample(Sample {
        op_name: "MatMul".into(),
        shape: vec![128, 128],
        dtype: DType::F32,
        device_chosen: DeviceTarget::GPU,
        duration_us: 200,
        vram_before: 0,
        vram_after: 0,
        fallback: false,
    });

    let info = make_info(vec![128, 128], DType::F32, DeviceTarget::GPU);

    let decision = selector.decide(&info);
    assert!(matches!(decision.prefer_device, Some(DeviceTarget::GPU)));
}

#[test]
fn adaptive_penalizes_gpu_when_fallbacks() {
    let mut selector = AdaptiveSelector::new();

    selector.register_sample(Sample {
        op_name: "MatMul".into(),
        shape: vec![256, 256],
        dtype: DType::F32,
        device_chosen: DeviceTarget::GPU,
        duration_us: 200,
        vram_before: 0,
        vram_after: 0,
        fallback: true,
    });

    selector.register_sample(Sample {
        op_name: "MatMul".into(),
        shape: vec![256, 256],
        dtype: DType::F32,
        device_chosen: DeviceTarget::CPU,
        duration_us: 300,
        vram_before: 0,
        vram_after: 0,
        fallback: false,
    });

    let info = make_info(vec![256, 256], DType::F32, DeviceTarget::CPU);

    let decision = selector.decide(&info);
    // Debido a los fallbacks en GPU, no deberíamos preferir explícitamente GPU.
    assert!(!matches!(decision.prefer_device, Some(DeviceTarget::GPU)));
}
