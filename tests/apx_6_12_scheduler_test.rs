mod mini_flux_common;

use atenia_engine::apx5::apx_5_3_planner::NodeExecInfo;
use atenia_engine::apx6_12::adaptive_scheduler::{
    AdaptiveScheduleBias,
    get_schedule_bias,
    set_schedule_bias,
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

fn run_miniflux_with_mode(mode: &str) -> Tensor {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", mode);
    }

    let cfg = default_cfg(2);
    let tokens = sample_tokens(&cfg, 0);
    run_logits_forward(&cfg, tokens)
}

#[test]
fn apx_6_12_default_bias_is_none() {
    // Asegurar estado limpio incluso si otros tests modificaron el bias.
    set_schedule_bias(AdaptiveScheduleBias::None);

    assert_eq!(get_schedule_bias(), AdaptiveScheduleBias::None);
}

#[test]
fn apx_6_12_can_set_bias() {
    set_schedule_bias(AdaptiveScheduleBias::QKVHeavy);
    assert_eq!(get_schedule_bias(), AdaptiveScheduleBias::QKVHeavy);
}

#[test]
fn apx_6_12_nodeexecinfo_bias_methods() {
    let mut info = NodeExecInfo::default();
    info.bias_attention_schedule();
    assert_eq!(info.exec_priority, Some(1));
}

#[test]
fn apx_6_12_mini_flux_outputs_are_unchanged() {
    let out_1 = run_miniflux_with_mode("6.11");
    let out_2 = run_miniflux_with_mode("6.12");

    let diff = max_abs_diff(&out_1, &out_2);
    assert!(
        diff < 1e-5,
        "MiniFlux outputs diverged between 6.11 and 6.12: diff={diff}",
    );
}
