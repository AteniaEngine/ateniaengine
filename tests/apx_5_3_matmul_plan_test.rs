use atenia_engine::apx5::apx_5_3_planner::{
    Planner5_3,
    NodeExecInfo,
    LayoutDecision,
};

fn make_info(num_elems: usize, dtype: &str, contiguous: bool) -> NodeExecInfo {
    // For this synthetic test we assume a 2D MatMul with shape [m, n]
    // such that m*n = num_elems. We use m=num_elems, n=1 for simplicity.
    NodeExecInfo {
        node_id: 0,
        op_name: "MatMul".to_string(),
        shape: vec![num_elems, 1],
        dtype: dtype.to_string(),
        contiguous,
        device_52: "CPU".to_string(),
        estimated_bytes: num_elems * 4,
        estimated_flops: num_elems,
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
fn small_matmul_prefers_cpu() {
    let planner = Planner5_3::new();
    // num_elems below the small threshold (4_096)
    let info = make_info(1024, "F32", true);

    let plan = planner.select_plan(&info);

    assert_eq!(plan.kernel_name, "matmul_cpu_small");
    matches!(plan.layout, LayoutDecision::Original);
}

#[test]
fn medium_matmul_cpu_medium() {
    let planner = Planner5_3::new();
    // Between small_threshold and large_threshold
    let info = make_info(100_000, "F32", true);

    let plan = planner.select_plan(&info);

    assert_eq!(plan.kernel_name, "matmul_cpu_medium");
    matches!(plan.layout, LayoutDecision::Original);
}

#[test]
fn large_contiguous_f32_marked_gpu_candidate() {
    let planner = Planner5_3::new();
    // Large, contiguous, and F32: should be marked as a GPU candidate
    let info = make_info(2_000_000, "F32", true);

    let plan = planner.select_plan(&info);

    assert_eq!(plan.kernel_name, "matmul_gpu_candidate");
    matches!(plan.layout, LayoutDecision::Original);
}

#[test]
fn large_non_contiguous_suggests_force_contiguous() {
    let planner = Planner5_3::new();
    // Large and non-contiguous: suggests ForceContiguous in the plan
    let info = make_info(2_000_000, "F32", false);

    let plan = planner.select_plan(&info);

    assert!(matches!(plan.layout, LayoutDecision::ForceContiguous));
}
