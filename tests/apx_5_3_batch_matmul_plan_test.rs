use atenia_engine::apx5::apx_5_3_planner::{Planner5_3, NodeExecInfo, LayoutDecision};

fn make_info(shape: Vec<usize>, dtype: &str, contiguous: bool) -> NodeExecInfo {
    NodeExecInfo {
        node_id: 0,
        op_name: "BatchMatMul".to_string(),
        shape,
        dtype: dtype.to_string(),
        contiguous,
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
fn small_batch_matmul_prefers_cpu_small() {
    let planner = Planner5_3::new();
    let info = make_info(vec![1, 8, 8], "F32", true);

    let plan = planner.select_plan(&info);
    assert_eq!(plan.kernel_name, "batch_matmul_cpu_small");
    assert!(matches!(plan.layout, LayoutDecision::Original));
}

#[test]
fn medium_batch_matmul_prefers_cpu_medium() {
    let planner = Planner5_3::new();
    let info = make_info(vec![4, 64, 64], "F32", true);

    let plan = planner.select_plan(&info);
    assert_eq!(plan.kernel_name, "batch_matmul_cpu_medium");
    assert!(matches!(plan.layout, LayoutDecision::Original));
}

#[test]
fn large_contiguous_batch_matmul_gpu_candidate() {
    let planner = Planner5_3::new();
    let info = make_info(vec![8, 256, 256], "F32", true);

    let plan = planner.select_plan(&info);
    assert_eq!(plan.kernel_name, "batch_matmul_gpu_candidate");
    assert!(matches!(plan.layout, LayoutDecision::Original));
}

#[test]
fn large_non_contiguous_batch_matmul_force_contiguous() {
    let planner = Planner5_3::new();
    let info = make_info(vec![8, 256, 256], "F32", false);

    let plan = planner.select_plan(&info);
    assert_eq!(plan.kernel_name, "batch_matmul_gpu_candidate");
    assert!(matches!(plan.layout, LayoutDecision::ForceContiguous));
}
