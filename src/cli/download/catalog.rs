//! Curated model catalog for `atenia download`.
//!
//! The catalog is a small, hand-picked list of public, non-gated
//! Hugging Face checkpoints that Atenia can run today. It is
//! intentionally **not** a universal model index — it is a curated
//! shortcut for the common "first model" path. Adding a model here
//! means we have verified it loads end-to-end against the current
//! engine and CLI.
//!
//! Out of scope for v1 of the download command:
//!   - arbitrary `--hf-repo` downloads,
//!   - gated / private models (no OAuth, no tokens),
//!   - models that require accepting an extra licence click,
//!   - format auto-negotiation (each entry pins exactly one format).
//!
//! When the catalog needs to grow past ~30 entries this should
//! migrate to an embedded JSON/YAML manifest via `include_str!`;
//! today three entries fit cleanly as a `&'static` table.

/// On-disk format of the checkpoint. Mirrors the formats the engine
/// already supports today; expanding this enum is a separate engine
/// change, not a download-catalog change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFormat {
    Safetensors,
    Gguf,
}

impl ModelFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelFormat::Safetensors => "safetensors",
            ModelFormat::Gguf => "gguf",
        }
    }
}

/// One curated catalog entry. Field names are stable across releases:
/// scripts may filter by `alias`, `family`, `format` or `gated`.
#[derive(Clone, Copy, Debug)]
pub struct CatalogEntry {
    /// Short, kebab-case alias the user types. Must be unique.
    pub alias: &'static str,
    /// Pretty model name used in headers and `list` output.
    pub display_name: &'static str,
    /// Hugging Face repo id (`owner/repo`). The download builds
    /// per-file URLs as
    /// `https://huggingface.co/<hf_repo>/resolve/main/<file>`.
    pub hf_repo: &'static str,
    /// On-disk format the engine should expect after download.
    pub format: ModelFormat,
    /// Files to fetch from `<hf_repo>/resolve/main/`. Each file is
    /// downloaded sequentially to `<dest>/<file>.partial` and
    /// renamed atomically on completion.
    pub files: &'static [&'static str],
    /// Default subdirectory under `./models/` when the user omits
    /// `--dir`. Must be a single path component (no separators).
    pub default_subdir: &'static str,
    /// Approximate total download size, in megabytes. Informational
    /// only — used to set expectations in the pre-download header.
    pub approx_size_mb: u32,
    /// Whether the model is gated on Hugging Face. v1 only lists
    /// non-gated models; `gated = true` here means the catalog
    /// refuses to download and points the user at the manual flow.
    pub gated: bool,
    /// Family label for `list` output and the post-download
    /// "Next:" footer. Matches the families Atenia's adapters
    /// support today.
    pub family: &'static str,
    /// One-line rationale shown in the pre-download header (`notes`).
    pub notes: &'static str,
    /// Concrete command the user should run next after a successful
    /// download. Always includes the destination path.
    pub recommended_next: &'static str,
}

impl CatalogEntry {
    /// Base URL on Hugging Face for this entry. Used by `list` /
    /// `--dry-run` output and for building per-file URLs.
    pub fn source_url(&self) -> String {
        format!("https://huggingface.co/{}", self.hf_repo)
    }

    /// Per-file URL on Hugging Face. The `/resolve/main/` path has
    /// been stable since 2022 and serves both LFS-backed weights
    /// and small JSON files transparently.
    pub fn file_url(&self, file: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.hf_repo, file
        )
    }
}

/// The curated catalog itself. Three entries on purpose: one per
/// supported family that ships small, non-gated, fast-to-download
/// checkpoints suitable for a 5-minute first-time trial.
pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        alias: "smollm2-135m",
        display_name: "SmolLM2 135M Instruct",
        hf_repo: "HuggingFaceTB/SmolLM2-135M-Instruct",
        format: ModelFormat::Safetensors,
        files: &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        ],
        default_subdir: "smollm2-135m",
        approx_size_mb: 270,
        gated: false,
        family: "SmolLM",
        notes: "Smallest fully-validated checkpoint; ideal first download.",
        recommended_next:
            "atenia chat --model ./models/smollm2-135m",
    },
    CatalogEntry {
        alias: "tinyllama",
        display_name: "TinyLlama 1.1B Chat v1.0",
        hf_repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        format: ModelFormat::Safetensors,
        files: &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        ],
        default_subdir: "tinyllama-1.1b-chat",
        approx_size_mb: 2200,
        gated: false,
        family: "Llama",
        notes: "Compact Llama-architecture chat model.",
        recommended_next:
            "atenia chat --model ./models/tinyllama-1.1b-chat",
    },
    CatalogEntry {
        alias: "qwen2.5-0.5b",
        display_name: "Qwen2.5 0.5B Instruct",
        hf_repo: "Qwen/Qwen2.5-0.5B-Instruct",
        format: ModelFormat::Safetensors,
        files: &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
        ],
        default_subdir: "qwen2.5-0.5b-instruct",
        approx_size_mb: 1000,
        gated: false,
        family: "Qwen",
        notes: "Smallest Qwen2.5 chat checkpoint.",
        recommended_next:
            "atenia chat --model ./models/qwen2.5-0.5b-instruct",
    },
];

/// Look up a catalog entry by alias. Case-sensitive: aliases are
/// kebab-case by convention and the catalogue treats them as
/// canonical identifiers, not display strings.
pub fn find(alias: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.alias == alias)
}

/// Iterator over all catalog aliases. Used by the error renderer
/// to suggest known aliases when an unknown one is supplied.
pub fn aliases() -> impl Iterator<Item = &'static str> {
    CATALOG.iter().map(|e| e.alias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn aliases_are_unique() {
        let mut seen = HashSet::new();
        for entry in CATALOG {
            assert!(
                seen.insert(entry.alias),
                "duplicate alias `{}` in CATALOG",
                entry.alias
            );
        }
    }

    #[test]
    fn each_entry_lists_a_config_and_weights() {
        for entry in CATALOG {
            assert!(
                entry.files.iter().any(|f| *f == "config.json"),
                "entry {} is missing config.json",
                entry.alias
            );
            assert!(
                entry.files.iter().any(|f| {
                    f.ends_with(".safetensors") || f.ends_with(".gguf")
                }),
                "entry {} is missing a weight file",
                entry.alias
            );
        }
    }

    #[test]
    fn default_subdir_is_a_single_component() {
        for entry in CATALOG {
            assert!(
                !entry.default_subdir.contains('/')
                    && !entry.default_subdir.contains('\\'),
                "entry {} default_subdir must be a single component",
                entry.alias
            );
            assert!(!entry.default_subdir.is_empty());
        }
    }

    #[test]
    fn v1_catalog_excludes_gated_models() {
        for entry in CATALOG {
            assert!(
                !entry.gated,
                "v1 catalog must not list gated models; alias {}",
                entry.alias
            );
        }
    }

    #[test]
    fn find_returns_some_for_known_alias() {
        assert!(find("tinyllama").is_some());
        assert!(find("smollm2-135m").is_some());
        assert!(find("qwen2.5-0.5b").is_some());
    }

    #[test]
    fn find_returns_none_for_unknown_alias() {
        assert!(find("does-not-exist").is_none());
        assert!(find("TinyLlama").is_none(), "lookup is case-sensitive");
    }

    #[test]
    fn file_url_uses_resolve_main() {
        let e = find("tinyllama").unwrap();
        let u = e.file_url("config.json");
        assert_eq!(
            u,
            "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0\
             /resolve/main/config.json"
        );
    }
}
