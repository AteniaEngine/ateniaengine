#![allow(dead_code)]

use std::sync::Arc;

use crate::v15::policy::policy::ExecutionPolicy;
use crate::v15::policy::registry::PolicyRegistry;

/// Orchestrates which execution policy is currently active.
///
/// The PolicyManager never evaluates policies, touches runtime state,
/// or accesses hardware. It only selects which policy should be used
/// by higher layers.
pub struct PolicyManager {
    registry: PolicyRegistry,
    active_name: String,
}

impl PolicyManager {
    /// Creates a new PolicyManager with the given registry and initial
    /// active policy name. The initial policy must exist in the
    /// registry; otherwise this function will panic.
    pub fn new(registry: PolicyRegistry, initial_policy: &str) -> Self {
        if registry.get(initial_policy).is_none() {
            panic!(
                "Initial policy '{}' not found in registry; PolicyManager requires a valid initial policy",
                initial_policy
            );
        }

        Self {
            registry,
            active_name: initial_policy.to_string(),
        }
    }

    /// Returns the name of the currently active policy.
    pub fn active_policy_name(&self) -> &str {
        &self.active_name
    }

    /// Returns a handle to the currently active policy.
    pub fn active_policy(&self) -> Arc<dyn ExecutionPolicy> {
        self
            .registry
            .get(&self.active_name)
            .expect("active policy must exist in registry")
    }

    /// Lists all available policy names in a deterministic order.
    pub fn list_available_policies(&self) -> Vec<&'static str> {
        self.registry.list()
    }

    /// Attempts to switch the active policy.
    ///
    /// If the target policy does not exist, this method returns false
    /// and leaves the active policy unchanged.
    pub fn set_active_policy(&mut self, name: &str) -> bool {
        if self.registry.get(name).is_some() {
            self.active_name = name.to_string();
            true
        } else {
            false
        }
    }
}
