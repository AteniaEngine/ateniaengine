//! Weight mapper (M4-c).
//!
//! Formalizes the "safetensors tensor name → graph parameter node_id"
//! mapping that M4-b validated ad-hoc via `HashMap::from_iter`, and
//! adds the validations that the raw iterator approach did not perform:
//!
//! - **Shape**: every tensor loaded from the safetensors file must
//!   match the shape of the parameter tensor already materialized in
//!   the graph at build time. Mismatch surfaces as
//!   [`LoaderError::ShapeMismatch`] with both shapes attached.
//! - **Dtype**: only F32 is supported in M4-c. BF16 / F16 / FP8 fall
//!   through as [`LoaderError::UnsupportedDType`] via the M4-a
//!   `TensorEntry::to_vec_f32` path. M4-d extends decode coverage.
//!
//! Mismatches between the mapper's name set and the reader's name
//! set are **not** fatal in M4-c. `load_into` reports them through
//! [`LoadReport::skipped`] (tensors in the file but not in the
//! mapper — e.g. producer metadata tensors) and [`LoadReport::missing`]
//! (tensors the mapper expected but the file did not provide —
//! e.g. partial checkpoints). Callers decide whether to treat them
//! as errors. A future strict-mode entry point can be added without
//! changing the loose-mode surface.
//!
//! # Design notes
//!
//! The mapper is intentionally decoupled from `MiniFluxHandles` or
//! any specific architecture struct. Its constructor accepts two
//! index-aligned slices (names + node_ids) that any `build_*` helper
//! in `src/nn/` can produce. When TinyLlama (M4.5) or other model
//! builders land, they reuse the same mapper without needing new
//! `from_xyz_handles` convenience constructors.

use std::collections::HashMap;

use crate::amg::graph::Graph;

use super::loader_errors::LoaderError;
use super::safetensors_reader::SafetensorsReader;

/// Bidirectional view from a tensor's logical name (as used inside
/// `register_weight`-style builders) to the graph parameter node that
/// holds its values. Construct with
/// [`WeightMapper::from_param_names_and_ids`].
#[derive(Debug, Clone)]
pub struct WeightMapper {
    mapping: HashMap<String, usize>,
}

/// Summary returned by [`WeightMapper::load_into`] after a
/// loose-mode load. `skipped` and `missing` are informational in
/// M4-c; callers can choose to enforce emptiness at their own layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Count of tensors that were matched, validated, and whose
    /// values were copied into the corresponding graph parameter.
    pub loaded: usize,
    /// Names present in the safetensors file but absent from the
    /// mapper. Typical cause: checkpoint ships extra metadata tensors
    /// or auxiliary tables (tokenizer embeddings, EMA shadows) that
    /// the current graph does not consume.
    pub skipped: Vec<String>,
    /// Names present in the mapper but absent from the file. Typical
    /// cause: sharded or partial checkpoint. Fatal in strict-mode
    /// scenarios, informational in the loose-mode `load_into`.
    pub missing: Vec<String>,
}

impl WeightMapper {
    /// Build a mapper from two index-aligned slices. Fails with
    /// [`LoaderError::InvalidFormat`] if the slices have different
    /// lengths, or if the name slice has duplicates (the resulting
    /// `HashMap` would silently drop collisions — surfacing the
    /// duplicate as an error is preferable to a hard-to-debug load).
    pub fn from_param_names_and_ids(
        names: &[String],
        ids: &[usize],
    ) -> Result<Self, LoaderError> {
        if names.len() != ids.len() {
            return Err(LoaderError::InvalidFormat(format!(
                "WeightMapper: names.len()={} differs from ids.len()={}; \
                 the two slices must be index-aligned",
                names.len(),
                ids.len()
            )));
        }

        let mut mapping: HashMap<String, usize> = HashMap::with_capacity(names.len());
        for (name, id) in names.iter().zip(ids.iter()) {
            if mapping.insert(name.clone(), *id).is_some() {
                return Err(LoaderError::InvalidFormat(format!(
                    "WeightMapper: duplicate parameter name '{}'; \
                     a safetensors checkpoint cannot disambiguate a \
                     duplicated name, so refusing to build a mapper \
                     that would silently prefer one node over another",
                    name
                )));
            }
        }

        Ok(Self { mapping })
    }

    /// Number of entries in the mapper.
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    /// Whether the mapper contains a given parameter name.
    pub fn contains(&self, name: &str) -> bool {
        self.mapping.contains_key(name)
    }

    /// Look up a parameter name. Returns the node_id of the graph
    /// parameter that holds its tensor, or `None` if the name is not
    /// mapped.
    pub fn get(&self, name: &str) -> Option<usize> {
        self.mapping.get(name).copied()
    }

    /// Load every tensor from `reader` whose name is present in the
    /// mapper into the corresponding graph parameter node. Validates
    /// shape on each load; surfaces dtype-unsupported cases from the
    /// reader as-is.
    ///
    /// Loose-mode: missing entries (in mapper, absent from reader)
    /// and skipped entries (in reader, absent from mapper) are
    /// reported through [`LoadReport`] rather than raising errors.
    /// Shape mismatch and dtype-unsupported are hard errors — the
    /// load aborts on the first occurrence.
    pub fn load_into(
        &self,
        graph: &mut Graph,
        reader: &SafetensorsReader,
    ) -> Result<LoadReport, LoaderError> {
        let mut report = LoadReport::default();

        // Names that the reader exposed; used to detect `missing`
        // entries (in mapper, absent from reader).
        let reader_names: std::collections::HashSet<String> =
            reader.iter().map(|e| e.name.to_string()).collect();

        for entry in reader.iter() {
            let Some(&node_id) = self.mapping.get(entry.name) else {
                report.skipped.push(entry.name.to_string());
                continue;
            };

            let tensor = graph
                .nodes
                .get_mut(node_id)
                .and_then(|n| n.output.as_mut())
                .ok_or_else(|| {
                    LoaderError::InvalidFormat(format!(
                        "mapper points '{}' at node_id {} but that node has no \
                         materialized tensor in the graph",
                        entry.name, node_id
                    ))
                })?;

            // Shape must match exactly. The safetensors format carries
            // the shape as `Vec<usize>`; the graph parameter carries
            // it in `Tensor::shape`. No broadcasting, no squeezing —
            // a mismatch means the source checkpoint is for a
            // different architecture configuration, and silent
            // acceptance would corrupt inference.
            if entry.shape != tensor.shape.as_slice() {
                return Err(LoaderError::ShapeMismatch {
                    tensor_name: entry.name.to_string(),
                    expected: tensor.shape.clone(),
                    actual: entry.shape.to_vec(),
                });
            }

            // Decode values. Propagates `UnsupportedDType` for BF16/
            // F16/FP8 unchanged from the reader (M4-d extends the
            // decode path, not the mapper).
            let values = entry.to_vec_f32()?;

            // Copy in-place into the existing CPU storage. Keeps
            // shape, layout, strides, and storage variant intact —
            // only the numeric values change.
            let slice = tensor.as_cpu_slice_mut();
            if slice.len() != values.len() {
                // This branch is defensive: the shape-match check
                // above already implies matching element counts, but
                // keeping an explicit guard makes the copy_from_slice
                // panic path impossible to reach and the diagnostic
                // clearer if the invariant is ever broken.
                return Err(LoaderError::InvalidFormat(format!(
                    "tensor '{}': element count {} after decode does not match \
                     graph parameter capacity {}",
                    entry.name,
                    values.len(),
                    slice.len()
                )));
            }
            slice.copy_from_slice(&values);
            report.loaded += 1;
        }

        // Detect missing entries: names the mapper expected to fill
        // that the reader did not provide.
        for name in self.mapping.keys() {
            if !reader_names.contains(name) {
                report.missing.push(name.clone());
            }
        }

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_param_names_and_ids_rejects_length_mismatch() {
        let names = vec!["a".to_string(), "b".to_string()];
        let ids = vec![1usize];
        let err = WeightMapper::from_param_names_and_ids(&names, &ids)
            .expect_err("length mismatch must fail");
        match err {
            LoaderError::InvalidFormat(msg) => {
                assert!(msg.contains("index-aligned"));
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn from_param_names_and_ids_rejects_duplicates() {
        let names = vec!["dup".to_string(), "dup".to_string()];
        let ids = vec![1usize, 2usize];
        let err = WeightMapper::from_param_names_and_ids(&names, &ids)
            .expect_err("duplicate name must fail");
        match err {
            LoaderError::InvalidFormat(msg) => {
                assert!(msg.contains("duplicate"));
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn from_param_names_and_ids_accepts_empty() {
        let mapper = WeightMapper::from_param_names_and_ids(&[], &[])
            .expect("empty is legal (zero-param model)");
        assert_eq!(mapper.len(), 0);
        assert!(!mapper.contains("anything"));
        assert_eq!(mapper.get("anything"), None);
    }
}
