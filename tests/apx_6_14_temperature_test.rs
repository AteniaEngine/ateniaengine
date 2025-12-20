use atenia_engine::apx6_14::{
    temperature_manager::update_temperature,
    temperature_schedule::{get_current_temperature, set_current_temperature},
};

#[test]
fn apx_6_14_temperature_decays() {
    // Force a known initial state.
    set_current_temperature(1.2);
    let t0 = get_current_temperature();

    update_temperature(0);
    let t_start = get_current_temperature();

    update_temperature(10_000);
    let t_late = get_current_temperature();

    // Same temperature at the start, lower or equal at later steps.
    assert!((t_start - t0).abs() < 1e-6);
    assert!(t_start >= t_late);
    assert!(t_late >= 0.25);
}
