use atenia_engine::apx6_10::FusionSelector;
use atenia_engine::apx6_10::FusionProfile;

#[test]
fn selector_prefers_full_when_much_faster() {
    let mut sel = FusionSelector::new();
    sel.record_profile(FusionProfile {
        op_name: "Attn".to_string(),
        baseline_us: 1000,
        fused_qkv_us: 900,
        fused_full_us: 600,
    });

    let decision = sel.decide();
    assert_eq!(decision.use_full_fusion, Some(true));
}

#[test]
fn selector_prefers_qkv_when_full_slower() {
    let mut sel = FusionSelector::new();
    sel.record_profile(FusionProfile {
        op_name: "Attn".to_string(),
        baseline_us: 1000,
        fused_qkv_us: 600,
        fused_full_us: 900,
    });

    let decision = sel.decide();
    assert_eq!(decision.use_full_fusion, Some(false));
}

#[test]
fn selector_returns_none_without_data() {
    let sel = FusionSelector::new();
    let decision = sel.decide();
    assert_eq!(decision.use_full_fusion, None);
}

#[test]
fn selector_returns_none_when_close() {
    let mut sel = FusionSelector::new();
    sel.record_profile(FusionProfile {
        op_name: "Attn".to_string(),
        baseline_us: 1000,
        fused_qkv_us: 800,
        fused_full_us: 790,
    });

    let decision = sel.decide();
    assert_eq!(decision.use_full_fusion, None);
}
