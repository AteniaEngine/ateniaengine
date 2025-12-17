use atenia_engine::ApxStabilizer;
use atenia_engine::ApxTemperature;
use atenia_engine::apx6_10::GlobalDecision;

#[test]
fn apx_6_15_stabilizes_to_expected_policy() {
    let mut stab = ApxStabilizer::new();
    let temp = ApxTemperature::from_value(0.5);

    let d1 = stab.stabilize(Some(GlobalDecision::PreferFull), &temp);
    let d2 = stab.stabilize(Some(GlobalDecision::PreferFull), &temp);

    assert!(matches!(d1, GlobalDecision::PreferFull));
    assert!(matches!(d2, GlobalDecision::PreferFull));
}

#[test]
fn apx_6_15_fallsback_when_no_data() {
    let mut stab = ApxStabilizer::new();
    let temp = ApxTemperature::from_value(1.0);

    let d = stab.stabilize(None, &temp);
    assert!(matches!(d, GlobalDecision::NoPreference));
}
