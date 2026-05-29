//! **MOE-14** — Real local MoE checkpoint smoke harness (experimental,
//! CPU-only, opt-in, out-of-CI).
//!
//! MOE-13 validated the metadata → stack → forward pipeline against a
//! synthetic-but-real-format checkpoint built in memory. MOE-14 adds the
//! plumbing to point that same pipeline at a **real MoE checkpoint sitting in
//! a local directory** (no download, no model in the repo): discover the
//! `.safetensors` shards, read a minimal slice of `config.json` if present,
//! merge the shards into one `MoeWeightMap` + byte resolver, and feed it to
//! the MOE-13 `RealMoeCheckpointValidation`.
//!
//! ## Critical scope notes
//!
//! * Everything here is for an **opt-in, `#[ignore]`d smoke test** gated on
//!   the `ATENIA_MOE_REAL_MODEL` env var. Nothing runs in CI by default.
//! * It **never downloads** anything and assumes no HuggingFace online access
//!   — it only reads files already on disk.
//! * It does **not** touch the productive loader load-path, the Adapter
//!   Toolkit, the CLI, generation, CUDA/ROCm/Metal, or the tier planner. It
//!   reuses the `SafetensorsReader`'s **public** read API only, exactly as the
//!   MOE-9..13 integration tests do, so the MOE-2 fail-loud guard is
//!   unchanged — a real MoE checkpoint still refuses to load as a model.
//! * A smoke PASS means "the experimental MoE path read this checkpoint,
//!   assembled a stack, and ran a finite forward", NOT "this model is
//!   correct, supported, or certified". No real Mixtral / Qwen-MoE support is
//!   claimed.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::data_plane::MoeWeightMap;
use crate::v17::loader::safetensors_reader::SafetensorsReader;

/// Errors from the local smoke harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoeSmokeError {
    /// The model directory does not exist.
    DirNotFound(String),
    /// The path exists but is not a directory.
    NotADirectory(String),
    /// No `.safetensors` files were found in the directory.
    NoSafetensors(String),
    /// A filesystem error occurred.
    Io(String),
    /// `config.json` was requested but not present.
    ConfigNotFound(String),
    /// `config.json` was present but could not be parsed as JSON.
    ConfigParse(String),
    /// A reader failed to open / parse a shard.
    Reader(String),
}

impl std::fmt::Display for MoeSmokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeSmokeError::DirNotFound(p) => write!(f, "moe-smoke: model dir not found: {p}"),
            MoeSmokeError::NotADirectory(p) => write!(f, "moe-smoke: not a directory: {p}"),
            MoeSmokeError::NoSafetensors(p) => {
                write!(f, "moe-smoke: no .safetensors files in {p}")
            }
            MoeSmokeError::Io(e) => write!(f, "moe-smoke: io error: {e}"),
            MoeSmokeError::ConfigNotFound(p) => write!(f, "moe-smoke: config.json not found in {p}"),
            MoeSmokeError::ConfigParse(e) => write!(f, "moe-smoke: config.json parse error: {e}"),
            MoeSmokeError::Reader(e) => write!(f, "moe-smoke: reader error: {e}"),
        }
    }
}

impl std::error::Error for MoeSmokeError {}

/// Discover all `.safetensors` files in `model_dir`, sorted by filename so
/// shards (`model-00001-of-00003.safetensors`, ...) come in order.
///
/// Pure filesystem listing — opens nothing, downloads nothing. Errors if the
/// directory is missing, is not a directory, or contains no `.safetensors`.
pub fn discover_safetensors_files(model_dir: &Path) -> Result<Vec<PathBuf>, MoeSmokeError> {
    if !model_dir.exists() {
        return Err(MoeSmokeError::DirNotFound(model_dir.display().to_string()));
    }
    if !model_dir.is_dir() {
        return Err(MoeSmokeError::NotADirectory(model_dir.display().to_string()));
    }
    let mut files: Vec<PathBuf> = Vec::new();
    let entries = fs::read_dir(model_dir).map_err(|e| MoeSmokeError::Io(e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| MoeSmokeError::Io(e.to_string()))?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        {
            files.push(path);
        }
    }
    if files.is_empty() {
        return Err(MoeSmokeError::NoSafetensors(model_dir.display().to_string()));
    }
    // Deterministic shard order.
    files.sort();
    Ok(files)
}

/// A minimal slice of a model's `config.json` — only the fields relevant to
/// assembling a MoE stack. Every field is optional: a real config may use a
/// different key, or `config.json` may be absent entirely.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MinimalMoeConfig {
    pub num_hidden_layers: Option<usize>,
    pub num_experts: Option<usize>,
    pub num_experts_per_token: Option<usize>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
}

impl MinimalMoeConfig {
    /// Parse the minimal MoE fields from a `config.json` JSON string.
    ///
    /// Accepts the common key spellings across families:
    /// - experts: `num_experts` (Qwen-MoE/DeepSeek) or `num_local_experts`
    ///   (Mixtral);
    /// - top-k: `num_experts_per_tok` or `num_experts_per_token`.
    ///
    /// This is **not** a full config parser — unknown fields are ignored and
    /// missing fields stay `None`. No Adapter Toolkit involvement.
    pub fn from_json_str(s: &str) -> Result<Self, MoeSmokeError> {
        let v: Value = serde_json::from_str(s).map_err(|e| MoeSmokeError::ConfigParse(e.to_string()))?;
        let as_usize = |key: &str| v.get(key).and_then(|x| x.as_u64()).map(|n| n as usize);
        let first = |keys: &[&str]| keys.iter().find_map(|k| as_usize(k));
        Ok(Self {
            num_hidden_layers: as_usize("num_hidden_layers"),
            num_experts: first(&["num_experts", "num_local_experts"]),
            num_experts_per_token: first(&["num_experts_per_tok", "num_experts_per_token"]),
            hidden_size: as_usize("hidden_size"),
            intermediate_size: as_usize("intermediate_size"),
        })
    }

    /// Read + parse `config.json` from a model directory. Returns
    /// `ConfigNotFound` if the file is absent (the caller may fall back).
    pub fn from_dir(model_dir: &Path) -> Result<Self, MoeSmokeError> {
        let path = model_dir.join("config.json");
        if !path.exists() {
            return Err(MoeSmokeError::ConfigNotFound(model_dir.display().to_string()));
        }
        let text = fs::read_to_string(&path).map_err(|e| MoeSmokeError::Io(e.to_string()))?;
        Self::from_json_str(&text)
    }

    /// The routing top-k to use, preferring the parsed config and falling
    /// back to `default` (used when `config.json` is absent or omits it).
    pub fn experts_per_token_or(&self, default: usize) -> usize {
        self.num_experts_per_token.unwrap_or(default)
    }

    /// `true` if no MoE-relevant field was found (config present but not a
    /// recognisable MoE config, or all keys differ).
    pub fn is_unknown_topology(&self) -> bool {
        self.num_experts.is_none() && self.num_experts_per_token.is_none()
    }
}

/// A real local MoE checkpoint opened for the smoke harness: owns one
/// `SafetensorsReader` per shard and exposes a merged metadata map + a byte
/// resolver across all shards.
pub struct LocalMoeCheckpoint {
    readers: Vec<SafetensorsReader>,
}

impl LocalMoeCheckpoint {
    /// Open every shard file into a reader. Reads file bytes (no mmap, no
    /// download). Fails loud (as a `Reader` error) on a malformed shard.
    pub fn open(files: &[PathBuf]) -> Result<Self, MoeSmokeError> {
        let mut readers = Vec::with_capacity(files.len());
        for f in files {
            let r = SafetensorsReader::open(f).map_err(|e| MoeSmokeError::Reader(e.to_string()))?;
            readers.push(r);
        }
        Ok(Self { readers })
    }

    /// Number of shards opened.
    pub fn num_shards(&self) -> usize {
        self.readers.len()
    }

    /// Total tensors across all shards.
    pub fn num_tensors(&self) -> usize {
        self.readers.iter().map(|r| r.len()).sum()
    }

    /// Build a merged `MoeWeightMap` from every shard's `(name, shape)`.
    pub fn weight_map(&self) -> MoeWeightMap {
        let pairs: Vec<(String, Vec<usize>)> = self
            .readers
            .iter()
            .flat_map(|r| r.iter().map(|e| (e.name.to_string(), e.shape.to_vec())))
            .collect();
        MoeWeightMap::from_tensors(pairs.iter().map(|(n, s)| (n.as_str(), s.clone())))
    }

    /// Resolve one tensor's f32 data, searching every shard. Suitable as the
    /// MOE-10/13 byte resolver: `|name| ckpt.resolve(name)`.
    pub fn resolve(&self, name: &str) -> Option<Vec<f32>> {
        self.readers
            .iter()
            .find_map(|r| r.get(name).and_then(|e| e.to_vec_f32().ok()))
    }
}

// ============================================================================
// Lightweight tests (real temp dirs + tiny JSON — NO real models)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp dir for a test (process id keeps parallel runs apart).
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("atenia_moe14_{}_{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(dir: &Path, name: &str) {
        fs::write(dir.join(name), b"").unwrap();
    }

    #[test]
    fn discover_safetensors_files_empty_dir() {
        let dir = temp_dir("empty");
        let err = discover_safetensors_files(&dir).unwrap_err();
        assert!(matches!(err, MoeSmokeError::NoSafetensors(_)));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discover_safetensors_files_missing_dir() {
        let dir = std::env::temp_dir().join("atenia_moe14_definitely_absent_dir_xyz");
        let _ = fs::remove_dir_all(&dir);
        let err = discover_safetensors_files(&dir).unwrap_err();
        assert!(matches!(err, MoeSmokeError::DirNotFound(_)));
    }

    #[test]
    fn discover_safetensors_files_single_file() {
        let dir = temp_dir("single");
        touch(&dir, "model.safetensors");
        touch(&dir, "config.json"); // ignored by discovery
        touch(&dir, "tokenizer.json"); // ignored
        let files = discover_safetensors_files(&dir).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_name().unwrap(), "model.safetensors");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discover_safetensors_files_shards_sorted() {
        let dir = temp_dir("shards");
        // Create out of order; discovery must return sorted.
        touch(&dir, "model-00003-of-00003.safetensors");
        touch(&dir, "model-00001-of-00003.safetensors");
        touch(&dir, "model-00002-of-00003.safetensors");
        let files = discover_safetensors_files(&dir).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(
            names,
            vec![
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            ]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_parse_qwen_moe_fields() {
        let json = r#"{
            "num_hidden_layers": 24,
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "hidden_size": 2048,
            "intermediate_size": 5632
        }"#;
        let cfg = MinimalMoeConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.num_hidden_layers, Some(24));
        assert_eq!(cfg.num_experts, Some(60));
        assert_eq!(cfg.num_experts_per_token, Some(4));
        assert_eq!(cfg.hidden_size, Some(2048));
        assert_eq!(cfg.intermediate_size, Some(5632));
        assert!(!cfg.is_unknown_topology());
        assert_eq!(cfg.experts_per_token_or(2), 4);
    }

    #[test]
    fn config_parse_mixtral_field_spellings() {
        // Mixtral uses num_local_experts + num_experts_per_tok.
        let json = r#"{ "num_local_experts": 8, "num_experts_per_tok": 2, "hidden_size": 4096 }"#;
        let cfg = MinimalMoeConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.num_experts, Some(8));
        assert_eq!(cfg.num_experts_per_token, Some(2));
    }

    #[test]
    fn config_parse_missing_fields_reports_error() {
        // A dense (non-MoE) config: parses fine, but no MoE topology.
        let json = r#"{ "num_hidden_layers": 32, "hidden_size": 4096 }"#;
        let cfg = MinimalMoeConfig::from_json_str(json).unwrap();
        assert!(cfg.is_unknown_topology());
        assert_eq!(cfg.num_experts, None);
        assert_eq!(cfg.num_experts_per_token, None);
        // Falls back to the supplied default.
        assert_eq!(cfg.experts_per_token_or(2), 2);
    }

    #[test]
    fn config_parse_invalid_json_errors() {
        let err = MinimalMoeConfig::from_json_str("{ not valid json").unwrap_err();
        assert!(matches!(err, MoeSmokeError::ConfigParse(_)));
    }

    #[test]
    fn config_from_dir_missing_is_reported() {
        let dir = temp_dir("noconfig");
        let err = MinimalMoeConfig::from_dir(&dir).unwrap_err();
        assert!(matches!(err, MoeSmokeError::ConfigNotFound(_)));
        let _ = fs::remove_dir_all(&dir);
    }
}
