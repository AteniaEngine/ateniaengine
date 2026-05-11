#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use super::policy::ExecutionPolicy;

pub struct PolicyRegistry {
    policies: HashMap<&'static str, Arc<dyn ExecutionPolicy>>,
}

impl PolicyRegistry {
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
        }
    }

    pub fn register<P>(&mut self, policy: P)
    where
        P: ExecutionPolicy + 'static,
    {
        let name = policy.name();
        self.policies.insert(name, Arc::new(policy));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ExecutionPolicy>> {
        self.policies.get(name).cloned()
    }

    pub fn list(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.policies.keys().copied().collect();
        names.sort_unstable();
        names
    }
}
