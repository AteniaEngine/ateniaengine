//! M10.3.1.0 — minimal parser for the per-checkpoint numeric
//! certification manifest (`model.numcert.json`).
//!
//! Schema documented in `docs/CERTIFICATION.md` (v1.0.0). This
//! module reads only the fields required for runtime mode
//! selection — the rest of the manifest is documentation /
//! audit data and is not parsed (the manifest is published to
//! be read by humans and external tools as well as by the
//! engine).
//!
//! The per-tensor policy field is **reserved** in v1 (always
//! `null`) and ignored here; M10.3.1.1 will extend the parser
//! when v2.0.0 of the schema introduces a populated
//! `per_tensor_policy` block.
//!
//! ## Lookup contract
//!
//! [`load_manifest`] returns:
//! - `Some(manifest)` if `<model_dir>/model.numcert.json` exists
//!   and parses cleanly.
//! - `None` if the file is absent (no manifest = no claim — the
//!   pipeline falls back to `ATENIA_FAST_MODE` env semantics).
//! - `None` with a stderr warning if the file is present but
//!   malformed (operator presumably mis-edited the JSON; the
//!   safe path is to ignore and fall back, not to abort the
//!   load).
//!
//! ## Mode resolution priority
//!
//! Callers (today only `pipeline::from_model_dir_with_options`)
//! apply this order:
//!
//! 1. `ATENIA_FAST_MODE=1` env var — operator override, wins
//!    unconditionally.
//! 2. `manifest.recommended_mode` — checkpoint-default chosen
//!    by the certification author.
//! 3. `MatmulMode::Certified` — safe fallback when neither of
//!    the above produced a value.

use std::fs;
use std::path::{Path, PathBuf};

/// Execution mode selected per matmul / per process. M10.3.1.0
/// uses a single value across the whole forward; M10.3.1.1
/// extends to per-tensor selection driven by
/// `manifest.per_tensor_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulMode {
    /// `cuda_matmul_bf16_inplace` (Path B M8.4c +
    /// `CUBLAS_COMPUTE_32F_FAST_TF32`). Satisfies ADR-004 strict
    /// on every model in the M4.6 fixture.
    Certified,
    /// `cuda_matmul_bf16_native_inplace` (BF16-TC native via
    /// `cublasGemmEx(BF16, BF16, F32)`). Industry-standard drift
    /// profile; ADR-005 envelope.
    Fast,
}

/// Parsed view of the manifest. v1.0.0 only exposes the fields
/// the engine consumes at runtime; the rest of the JSON is
/// preserved on disk for human / external-tool consumption.
#[derive(Debug, Clone)]
pub struct NumcertManifest {
    /// Schema version string from the manifest. Recorded so a
    /// future M10.3.1.1 (v2.0.0) can reject unknown majors and
    /// the v1.0.0 reader stays forward-compatible until then.
    pub schema_version: String,
    /// Mode the certification author recommends for production
    /// use of this checkpoint. The fixture data + rationale that
    /// justifies the choice live in the manifest's
    /// `recommended_mode_rationale` field on disk.
    pub recommended_mode: MatmulMode,
    /// Path the manifest was loaded from. Logged at load time
    /// so an operator running multiple checkpoints can see which
    /// file influenced the mode choice.
    pub source: PathBuf,
}

/// Canonical filename. The certification flow puts the manifest
/// next to `model.safetensors[.index.json]`.
pub const MANIFEST_FILENAME: &str = "model.numcert.json";

/// Read the manifest at `<model_dir>/model.numcert.json`.
///
/// Returns `None` (without error) when the file is absent — a
/// missing manifest is the documented zero state, not a failure.
/// Returns `None` with a stderr warning when the file exists but
/// fails to parse — the safe behaviour is to ignore the broken
/// claim and fall back to `ATENIA_FAST_MODE` env semantics.
pub fn load_manifest(model_dir: &Path) -> Option<NumcertManifest> {
    let path = model_dir.join(MANIFEST_FILENAME);
    if !path.exists() {
        return None;
    }
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "[ATENIA] Numeric contract: failed to read {} ({}); \
                 honouring ATENIA_FAST_MODE only.",
                path.display(),
                e,
            );
            return None;
        }
    };
    match parse_manifest(&bytes, path.clone()) {
        Ok(m) => Some(m),
        Err(reason) => {
            eprintln!(
                "[ATENIA] Numeric contract: manifest at {} is malformed \
                 ({}); honouring ATENIA_FAST_MODE only.",
                path.display(),
                reason,
            );
            None
        }
    }
}

/// Pure-bytes parser. Split out so unit tests can exercise the
/// parsing without touching the filesystem. Returns either a
/// fully-resolved manifest or a one-line human-readable reason
/// for the failure (used by `load_manifest` to log).
fn parse_manifest(bytes: &[u8], source: PathBuf) -> Result<NumcertManifest, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| format!("not valid UTF-8: {e}"))?;

    let schema_version = extract_string_field(text, "schema_version")
        .ok_or_else(|| "missing or non-string \"schema_version\"".to_string())?;
    let recommended_mode_str = extract_string_field(text, "recommended_mode")
        .ok_or_else(|| "missing or non-string \"recommended_mode\"".to_string())?;
    let recommended_mode = match recommended_mode_str.as_str() {
        "certified" => MatmulMode::Certified,
        "fast" => MatmulMode::Fast,
        other => {
            return Err(format!(
                "unrecognised recommended_mode \"{other}\" (expected \"certified\" or \"fast\")"
            ));
        }
    };
    Ok(NumcertManifest {
        schema_version,
        recommended_mode,
        source,
    })
}

/// Minimal scanner: finds `"<name>": "<value>"` and returns
/// `<value>`. Tolerates whitespace around the colon. Does not
/// implement full JSON parsing — the manifest is small,
/// hand-authored, and we only consume two scalar fields here;
/// any future field will follow the same shape.
///
/// Returns `None` if the field is missing or not a string. The
/// caller turns that into the appropriate error message; the
/// stderr surface stays to `load_manifest`.
fn extract_string_field(text: &str, name: &str) -> Option<String> {
    let needle = format!("\"{name}\"");
    let key_pos = text.find(&needle)?;
    let after_key = &text[key_pos + needle.len()..];
    let colon = after_key.find(':')?;
    let after_colon = &after_key[colon + 1..];
    // Skip whitespace between `:` and `"`.
    let trimmed = after_colon.trim_start();
    if !trimmed.starts_with('"') {
        return None;
    }
    let after_open_quote = &trimmed[1..];
    // No escape handling needed: the values we read here
    // (schema_version, recommended_mode) are simple ASCII and
    // never contain quotes or backslashes.
    let close_quote = after_open_quote.find('"')?;
    Some(after_open_quote[..close_quote].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fake_source() -> PathBuf {
        PathBuf::from("test://memory")
    }

    #[test]
    fn load_manifest_returns_none_when_file_absent() {
        let dir = std::env::temp_dir().join("atenia_numcert_absent_test");
        let _ = std::fs::create_dir_all(&dir);
        // Make sure the manifest is not present.
        let _ = std::fs::remove_file(dir.join(MANIFEST_FILENAME));
        assert!(load_manifest(&dir).is_none());
    }

    #[test]
    fn parse_manifest_recognises_recommended_mode_fast() {
        let json = br#"{
            "schema_version": "1.0.0",
            "model": "Qwen2.5-1.5B-Instruct",
            "recommended_mode": "fast",
            "per_tensor_policy": null
        }"#;
        let m = parse_manifest(json, fake_source())
            .expect("manifest should parse");
        assert_eq!(m.schema_version, "1.0.0");
        assert_eq!(m.recommended_mode, MatmulMode::Fast);
    }

    #[test]
    fn parse_manifest_recognises_recommended_mode_certified() {
        let json = br#"{
            "schema_version": "1.0.0",
            "model": "TinyLlama-1.1B-Chat-v1.0",
            "recommended_mode": "certified",
            "per_tensor_policy": null
        }"#;
        let m = parse_manifest(json, fake_source())
            .expect("manifest should parse");
        assert_eq!(m.recommended_mode, MatmulMode::Certified);
    }

    #[test]
    fn parse_manifest_rejects_unknown_mode() {
        let json = br#"{
            "schema_version": "1.0.0",
            "recommended_mode": "turbo"
        }"#;
        let err = parse_manifest(json, fake_source())
            .expect_err("unknown mode should fail");
        assert!(err.contains("turbo"), "error should mention bad value, got: {err}");
    }

    #[test]
    fn parse_manifest_rejects_missing_recommended_mode() {
        let json = br#"{
            "schema_version": "1.0.0"
        }"#;
        let err = parse_manifest(json, fake_source())
            .expect_err("missing field should fail");
        assert!(err.contains("recommended_mode"), "error should name the missing field");
    }

    #[test]
    fn extract_string_field_handles_whitespace() {
        let text = r#"{ "key" :    "value", "other": "x" }"#;
        assert_eq!(extract_string_field(text, "key"), Some("value".to_string()));
        assert_eq!(extract_string_field(text, "other"), Some("x".to_string()));
        assert_eq!(extract_string_field(text, "missing"), None);
    }
}
