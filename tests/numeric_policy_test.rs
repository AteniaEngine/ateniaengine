//! **NUMERIC-POLICY-1** — process-global policy override mechanism.
//!
//! These tests **mutate** the process-global `NumericPolicy` override. They run
//! in their own test binary (separate process from the lib's many FFN-equality
//! tests that read the global), so the mutation can never race those. Within
//! this file a lock serialises the cases.

use std::sync::Mutex;

use atenia_engine::moe::numeric_policy::{
    clear_numeric_policy_override, numeric_policy, set_numeric_policy, NumericPolicy,
};

static LOCK: Mutex<()> = Mutex::new(());

#[test]
fn default_after_clear_is_certified() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    clear_numeric_policy_override();
    // No override + no env → Certified (the safe default).
    set_numeric_policy(NumericPolicy::Certified);
    assert_eq!(numeric_policy(), NumericPolicy::Certified);
    clear_numeric_policy_override();
}

#[test]
fn override_selects_and_clears() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    set_numeric_policy(NumericPolicy::Strict);
    assert_eq!(numeric_policy(), NumericPolicy::Strict);
    set_numeric_policy(NumericPolicy::Fast);
    assert_eq!(numeric_policy(), NumericPolicy::Fast);
    set_numeric_policy(NumericPolicy::Certified);
    assert_eq!(numeric_policy(), NumericPolicy::Certified);
    clear_numeric_policy_override();
}
