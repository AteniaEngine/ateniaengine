use atenia_engine::apx5_4::{AdaptiveSelector, Sample, DeviceTarget};
use atenia_engine::apx5::apx_5_3_planner::NodeExecInfo;
use atenia_engine::tensor::DType;

fn make_info(shape: Vec<usize>, dtype: DType) -> NodeExecInfo {
    NodeExecInfo {
        node_id: 0,
        op_name: "BatchMatMul".to_string(),
        shape,
        dtype: format!("{:?}", dtype),
        contiguous: true,
        device_52: "CPU".to_string(),
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
fn adaptive_prefers_cpu_for_batch_matmul_when_cpu_faster() {
    let mut selector = AdaptiveSelector::new();

    selector.register_sample(Sample {
        op_name: "BatchMatMul".into(),
        shape: vec![2, 64, 64],
        dtype: DType::F32,
        device_chosen: DeviceTarget::CPU,
        duration_us: 100,
        vram_before: 0,
        vram_after: 0,
        fallback: false,
    });

    selector.register_sample(Sample {
        op_name: "BatchMatMul".into(),
        shape: vec![2, 64, 64],
        dtype: DType::F32,
        device_chosen: DeviceTarget::GPU,
        duration_us: 300,
        vram_before: 0,
        vram_after: 0,
        fallback: false,
    });

    let info = make_info(vec![2, 64, 64], DType::F32);
    let pref = selector.device_preference_for(&info);
    assert!(matches!(pref, Some(DeviceTarget::CPU)));
}

#[test]
fn adaptive_prefers_gpu_for_batch_matmul_when_gpu_faster() {
    let mut selector = AdaptiveSelector::new();

    selector.register_sample(Sample {
        op_name: "BatchMatMul".into(),
        shape: vec![2, 128, 128],
        dtype: DType::F32,
        device_chosen: DeviceTarget::CPU,
        duration_us: 500,
        vram_before: 0,
        vram_after: 0,
        fallback: false,
    });

    selector.register_sample(Sample {
        op_name: "BatchMatMul".into(),
        shape: vec![2, 128, 128],
        dtype: DType::F32,
        device_chosen: DeviceTarget::GPU,
        duration_us: 200,
        vram_before: 0,
        vram_after: 0,
        fallback: false,
    });

    let info = make_info(vec![2, 128, 128], DType::F32);
    let pref = selector.device_preference_for(&info);
    assert!(matches!(pref, Some(DeviceTarget::GPU)));
}

#[test]
fn adaptive_returns_none_for_batch_matmul_when_no_data() {
    let selector = AdaptiveSelector::new();
    let info = make_info(vec![4, 256, 256], DType::F32);
    let pref = selector.device_preference_for(&info);
    assert!(pref.is_none());
}
