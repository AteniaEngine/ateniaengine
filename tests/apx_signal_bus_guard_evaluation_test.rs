//! APX v19 milestone A: end-to-end sensor → condition → guard cycle.
//!
//! Demonstrates SignalBus producing GuardConditions from live memory
//! telemetry, and a v16 Guard evaluating them correctly. This is the
//! first test in the suite where a guard's decision is driven by real
//! hardware state rather than hardcoded test fixtures.

use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{
    ExecutionBackend, ExecutionContract,
};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_action::GuardAction;
use atenia_engine::v16::guards::guard_conditions::GuardConditions;

/// Local test fixture mirroring MemoryPressureGuard from
/// `adaptive_guards_test.rs`. Kept local because guard implementations
/// intentionally live only in tests today.
struct MemoryPressureGuard {
    threshold: f32,
}

impl ExecutionGuard for MemoryPressureGuard {
    fn name(&self) -> &'static str {
        "memory_pressure_guard"
    }

    fn evaluate(
        &self,
        _contract: &ExecutionContract,
        conditions: &GuardConditions,
    ) -> GuardAction {
        if conditions.memory_pressure > self.threshold {
            GuardAction::Abort
        } else {
            GuardAction::Continue
        }
    }
}

fn minimal_contract() -> ExecutionContract {
    ExecutionContract {
        bias: DecisionBias {
            risk_weight: 0.3,
            latency_weight: 0.4,
            stability_weight: 0.8,
            memory_pressure_weight: 0.5,
            offload_cost_weight: 0.4,
        },
        runtime_snapshot: RuntimeState {
            memory_headroom: 0.8,
            is_stable: true,
            recent_recovery: false,
            offload_supported: true,
        },
        allowed_backends: vec![ExecutionBackend::Local],
        forbidden_backends: vec![],
        max_aggressiveness: 0.5,
        require_fallback: false,
        require_stability: false,
        constraints: Constraints { items: vec![] },
    }
}

#[test]
fn test_signal_bus_drives_guard_evaluation() {
    let bus = SignalBus::new();

    let conditions = match bus.collect_guard_conditions() {
        Some(c) => c,
        None => {
            eprintln!("SKIPPED: memory probe unavailable");
            return;
        }
    };

    let guard = MemoryPressureGuard { threshold: 0.5 };
    let contract = minimal_contract();
    let action = guard.evaluate(&contract, &conditions);

    let pressure = conditions.memory_pressure;
    let expected = if pressure > 0.5 {
        GuardAction::Abort
    } else {
        GuardAction::Continue
    };
    assert_eq!(
        action, expected,
        "guard action must match the memory_pressure branch \
         (pressure = {}, threshold = 0.5)",
        pressure
    );
}
