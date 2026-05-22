//! **Adapter Toolkit v2 — Part 4: the dynamic adapter registry.**
//!
//! [`AdapterRegistry`] holds v2 [`GeneratedAdapter`]s registered at
//! runtime from DSL files. Resolution is **v2-first, v1-fallback**:
//! a model's metadata is matched against the registered v2 adapters
//! first, and only if none match does it fall through to v1's
//! static [`resolve_adapter`]. v1 is never modified and never
//! shadowed for models that have no v2 spec, so back-compatibility
//! is total.
//!
//! Registered v2 adapters can be looked up three ways: by metadata
//! (the load-time path), by registration name (explicit selection),
//! or by family (override-by-family). This covers the "override por
//! familia / modelo / formato" requirement — `formato` is a
//! resolution input via [`ModelMetadata::format`], and `modelo` is
//! the per-checkpoint override layer carried inside each
//! [`GeneratedAdapter`]'s [`ResolvedAdapterSpec`].

use crate::model_adapters::{
    resolve_adapter, AteniaModelAdapter, ModelAdapter, ModelFamily, ModelMetadata,
};

use super::dsl::AdapterDsl;
use super::generator::GeneratedAdapter;
use super::spec::ResolvedAdapterSpec;
use super::ToolkitError;

/// The outcome of resolving a model against the registry.
pub enum AdapterResolution<'a> {
    /// A registered v2 adapter matched.
    V2 {
        /// The registration name of the matching v2 adapter.
        name: &'a str,
        /// The matching v2 adapter.
        adapter: &'a GeneratedAdapter,
    },
    /// No v2 adapter matched; v1's static registry resolved it.
    V1(&'static dyn AteniaModelAdapter),
    /// Neither layer could resolve the model.
    Unresolved,
}

impl AdapterResolution<'_> {
    /// `true` when a v2 adapter handled the resolution.
    pub fn is_v2(&self) -> bool {
        matches!(self, AdapterResolution::V2 { .. })
    }

    /// The resolved adapter's family, if any layer resolved it.
    pub fn family(&self) -> Option<ModelFamily> {
        match self {
            AdapterResolution::V2 { adapter, .. } => Some(adapter.family()),
            AdapterResolution::V1(a) => Some(a.family()),
            AdapterResolution::Unresolved => None,
        }
    }
}

struct RegistryEntry {
    name: String,
    adapter: GeneratedAdapter,
}

/// A runtime registry of v2 adapters with v1 fallback.
#[derive(Default)]
pub struct AdapterRegistry {
    entries: Vec<RegistryEntry>,
}

impl AdapterRegistry {
    /// An empty registry — every resolution falls straight through
    /// to v1 until something is registered.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a v2 adapter from an already-parsed DSL document.
    /// `name` is the lookup label (typically the DSL file stem).
    /// Re-registering an existing name replaces the entry.
    pub fn register_dsl(&mut self, name: &str, dsl: &AdapterDsl) -> Result<(), ToolkitError> {
        let spec = ResolvedAdapterSpec::resolve(dsl)?;
        let adapter = GeneratedAdapter::from_spec(spec)?;
        self.register_adapter(name, adapter);
        Ok(())
    }

    /// Register an already-built [`GeneratedAdapter`]. Replaces any
    /// entry with the same name.
    pub fn register_adapter(&mut self, name: &str, adapter: GeneratedAdapter) {
        if let Some(slot) = self.entries.iter_mut().find(|e| e.name == name) {
            slot.adapter = adapter;
        } else {
            self.entries.push(RegistryEntry {
                name: name.to_string(),
                adapter,
            });
        }
    }

    /// Number of registered v2 adapters.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no v2 adapter is registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The registration names, in insertion order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    /// Resolve a model: v2 adapters first (first registered match
    /// wins), then v1's static registry, then `Unresolved`.
    pub fn resolve(&self, metadata: &ModelMetadata<'_>) -> AdapterResolution<'_> {
        if let Some(entry) = self.entries.iter().find(|e| e.adapter.supports(metadata)) {
            return AdapterResolution::V2 {
                name: &entry.name,
                adapter: &entry.adapter,
            };
        }
        match resolve_adapter(metadata) {
            Some(v1) => AdapterResolution::V1(v1),
            None => AdapterResolution::Unresolved,
        }
    }

    /// Look up a registered v2 adapter by registration name.
    pub fn get(&self, name: &str) -> Option<&GeneratedAdapter> {
        self.entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| &e.adapter)
    }

    /// Look up the first registered v2 adapter for a family —
    /// the "override por familia" path.
    pub fn get_by_family(&self, family: ModelFamily) -> Option<&GeneratedAdapter> {
        self.entries
            .iter()
            .find(|e| e.adapter.family() == family)
            .map(|e| &e.adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_adapters::ModelFormat;

    fn dsl(text: &str) -> AdapterDsl {
        AdapterDsl::from_str(text, true).expect("dsl parses")
    }

    fn meta(arch: &'static str, model_type: &'static str) -> ModelMetadata<'static> {
        ModelMetadata {
            architecture: arch,
            model_type: Some(model_type),
            format: ModelFormat::HfSafetensors,
        }
    }

    #[test]
    fn empty_registry_falls_through_to_v1() {
        let reg = AdapterRegistry::new();
        let res = reg.resolve(&meta("LlamaForCausalLM", "llama"));
        assert!(!res.is_v2());
        assert_eq!(res.family(), Some(ModelFamily::Llama));
    }

    #[test]
    fn registered_v2_adapter_wins_over_v1() {
        let mut reg = AdapterRegistry::new();
        reg.register_dsl("my-llama", &dsl("family: llama\n"))
            .expect("registers");
        let res = reg.resolve(&meta("LlamaForCausalLM", "llama"));
        match res {
            AdapterResolution::V2 { name, adapter } => {
                assert_eq!(name, "my-llama");
                assert_eq!(adapter.family(), ModelFamily::Llama);
            }
            _ => panic!("expected v2 resolution"),
        }
    }

    #[test]
    fn v1_fallback_for_unregistered_family() {
        let mut reg = AdapterRegistry::new();
        reg.register_dsl("my-llama", &dsl("family: llama\n"))
            .expect("registers");
        // Phi has no v2 entry — must fall through to v1.
        let res = reg.resolve(&meta("Phi3ForCausalLM", "phi3"));
        assert!(!res.is_v2());
        assert_eq!(res.family(), Some(ModelFamily::Phi3));
    }

    #[test]
    fn unknown_architecture_is_unresolved() {
        let reg = AdapterRegistry::new();
        let res = reg.resolve(&meta("FalconForCausalLM", "falcon"));
        assert!(matches!(res, AdapterResolution::Unresolved));
        assert_eq!(res.family(), None);
    }

    #[test]
    fn re_registering_a_name_replaces_the_entry() {
        let mut reg = AdapterRegistry::new();
        reg.register_dsl("slot", &dsl("family: llama\n")).unwrap();
        reg.register_dsl("slot", &dsl("family: qwen\n")).unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(
            reg.get("slot").expect("present").family(),
            ModelFamily::Qwen2
        );
    }

    #[test]
    fn lookup_by_name_and_family() {
        let mut reg = AdapterRegistry::new();
        reg.register_dsl("a", &dsl("family: gemma3\n")).unwrap();
        reg.register_dsl("b", &dsl("family: mistral\n")).unwrap();
        assert!(reg.get("a").is_some());
        assert!(reg.get("missing").is_none());
        assert_eq!(
            reg.get_by_family(ModelFamily::Mistral)
                .expect("mistral present")
                .base_id(),
            "mistral"
        );
        assert!(reg.get_by_family(ModelFamily::Phi3).is_none());
    }

    #[test]
    fn names_iterates_in_insertion_order() {
        let mut reg = AdapterRegistry::new();
        reg.register_dsl("first", &dsl("family: llama\n")).unwrap();
        reg.register_dsl("second", &dsl("family: phi\n")).unwrap();
        let names: Vec<_> = reg.names().collect();
        assert_eq!(names, ["first", "second"]);
    }
}
