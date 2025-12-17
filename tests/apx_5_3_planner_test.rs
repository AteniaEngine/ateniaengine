use atenia_engine::apx5::apx_5_3_planner::{
    Planner5_3,
    NodeExecInfo,
    LayoutDecision,
};

#[test]
fn test_5_3_produces_coherent_plan() {
    let planner = Planner5_3::new();

    let info = NodeExecInfo {
        node_id: 0,
        op_name: "MatMul".to_string(),
        shape: vec![2048, 2048],
        dtype: "f16".to_string(),
        contiguous: true,
        device_52: "GPU".to_string(),
        estimated_bytes: 128 * 1024 * 1024,
        estimated_flops: 100_000_000,
        vram_free: 500 * 1024 * 1024,
        kernel_time_avg: 1.2,
        preferred_kernel_size: None,
        tile_override: None,
        scheduling_bias: None,
        qkv_bias: None,
        attention_bias: None,
        exec_priority: None,
        prefetch_hint: None,
    };

    let plan = planner.select_plan(&info);

    // Layout debe ser una de las decisiones válidas
    match plan.layout {
        LayoutDecision::Original
        | LayoutDecision::ForceContiguous
        | LayoutDecision::ForceChannelsFirst => {}
    }

    // Kernel name no vacío
    assert!(!plan.kernel_name.is_empty());
}
