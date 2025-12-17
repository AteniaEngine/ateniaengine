mod mini_flux_common;

use atenia_engine::apx5::apx_5_3_planner::NodeExecInfo;
use atenia_engine::apx6_11::runtime_policy::{
    FusionRuntimePolicy,
    get_runtime_policy,
    set_runtime_policy,
};
use atenia_engine::tensor::Tensor;
use mini_flux_common::{default_cfg, run_logits_forward, sample_tokens};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape, b.shape, "Tensors must have same shape to compare");
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, v| acc.max(v))
}

#[test]
fn apx_6_11_runtime_policy_default_and_set_get() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "6.11");
    }

    set_runtime_policy(FusionRuntimePolicy::Baseline);
    assert_eq!(get_runtime_policy(), FusionRuntimePolicy::Baseline);

    set_runtime_policy(FusionRuntimePolicy::PreferFull);
    assert_eq!(get_runtime_policy(), FusionRuntimePolicy::PreferFull);

    set_runtime_policy(FusionRuntimePolicy::PreferQKV);
    assert_eq!(get_runtime_policy(), FusionRuntimePolicy::PreferQKV);
}

#[test]
fn apx_6_11_node_exec_info_bias_methods() {
    let base_info = NodeExecInfo {
        node_id: 0,
        op_name: "MatMul".to_string(),
        shape: vec![16, 16],
        dtype: "F32".to_string(),
        contiguous: true,
        device_52: "CPU".to_string(),
        estimated_bytes: 16 * 16 * 4,
        estimated_flops: 16 * 16,
        vram_free: 0,
        kernel_time_avg: 0.0,
        preferred_kernel_size: None,
        tile_override: None,
        scheduling_bias: None,
        qkv_bias: None,
        attention_bias: None,
        exec_priority: None,
        prefetch_hint: None,
    };

    let mut qkv_info = base_info.clone();
    qkv_info.apply_qkv_bias();
    assert_eq!(qkv_info.preferred_kernel_size, Some((64, 64)));
    assert_eq!(qkv_info.scheduling_bias, Some("qkv_fastpath"));
    assert_eq!(qkv_info.qkv_bias, Some(true));
    assert_eq!(qkv_info.attention_bias, None);

    let mut attn_info = base_info.clone();
    attn_info.apply_attention_bias();
    assert_eq!(attn_info.preferred_kernel_size, Some((128, 128)));
    assert_eq!(attn_info.tile_override, Some((64, 128, 32)));
    assert_eq!(attn_info.scheduling_bias, Some("attn_large"));
    assert_eq!(attn_info.attention_bias, Some(true));
}

#[test]
fn apx_6_11_mini_flux_outputs_are_stable_across_policies() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "6.10");
    }

    let cfg = default_cfg(2);
    let tokens = sample_tokens(&cfg, 0);
    let logits_610 = run_logits_forward(&cfg, tokens.clone());

    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "6.11");
    }

    set_runtime_policy(FusionRuntimePolicy::PreferFull);
    let logits_full = run_logits_forward(&cfg, tokens.clone());

    set_runtime_policy(FusionRuntimePolicy::PreferQKV);
    let logits_qkv = run_logits_forward(&cfg, tokens);

    assert_eq!(logits_610.shape, logits_full.shape);
    assert_eq!(logits_610.shape, logits_qkv.shape);

    let diff_full = max_abs_diff(&logits_610, &logits_full);
    let diff_qkv = max_abs_diff(&logits_610, &logits_qkv);

    assert!(diff_full <= 1e-5, "6.10 vs 6.11 PreferFull diverged: diff={diff_full}");
    assert!(diff_qkv <= 1e-5, "6.10 vs 6.11 PreferQKV diverged: diff={diff_qkv}");
}
