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
    /// Quantized checkpoint path (GGUF Q4_K_M / Q8_0 / similar).
    /// This is a functional certification mode, not an ADR-004
    /// strict numerical mode. Runtime matmul precision falls back
    /// to the certified path; the distinction is carried so the
    /// manifest can state that quantization drift is intrinsic
    /// without being rejected as malformed.
    Quantized,
}

/// **M10.3.1.1** — per-tensor precision policy parsed from
/// schema v2.0.0 manifests. v1.0.0 manifests have
/// `per_tensor_policy: null` and produce `None` here, which
/// preserves the M10.3.1.0 whole-model contract.
#[derive(Debug, Clone)]
pub struct PerTensorPolicy {
    /// Mode used for any tensor name not matched by an override.
    pub default_mode: MatmulMode,
    /// Ordered list of (glob pattern, mode) overrides. Earlier
    /// entries take precedence over later ones for the same name.
    pub overrides: Vec<PolicyOverride>,
}

/// One override entry: a glob pattern and the mode it forces
/// when matched.
#[derive(Debug, Clone)]
pub struct PolicyOverride {
    /// Glob pattern. Supports a single `*` wildcard at the
    /// start, end, or middle of the pattern (suffix / prefix /
    /// contains match). Multi-`*` patterns are not supported by
    /// design — the manifests in the M4.6 fixture all fit the
    /// single-wildcard shape and richer matching is the wrong
    /// abstraction at v2.0.0 (re-evaluate at v3 if a real
    /// checkpoint forces it).
    pub pattern: String,
    /// Mode to force when `pattern` matches the tensor name.
    pub mode: MatmulMode,
}

impl PerTensorPolicy {
    /// Resolve the mode for a given tensor name. Walks the
    /// overrides in order; first match wins. Falls back to
    /// `default_mode` when no override matches.
    pub fn resolve(&self, tensor_name: &str) -> MatmulMode {
        for ov in &self.overrides {
            if glob_match(&ov.pattern, tensor_name) {
                return ov.mode;
            }
        }
        self.default_mode
    }
}

/// Parsed view of the manifest. The engine consumes
/// `recommended_mode` at load (M10.3.1.0) and `per_tensor_policy`
/// per-matmul (M10.3.1.1); the rest of the JSON is preserved on
/// disk for human / external-tool consumption.
#[derive(Debug, Clone)]
pub struct NumcertManifest {
    /// Schema version string from the manifest. v1.0.0 has
    /// `per_tensor_policy: null` and produces
    /// `per_tensor_policy = None` here. v2.0.0 introduced the
    /// populated `per_tensor_policy` block in M10.3.1.1.
    pub schema_version: String,
    /// Mode the certification author recommends for the whole
    /// model when no per-tensor data is available. Used by
    /// M10.3.1.0's `init_fast_mode_active` to set the global
    /// fallback.
    pub recommended_mode: MatmulMode,
    /// **M10.3.1.1** — per-tensor precision policy. `None` for
    /// v1.0.0 manifests (fall back to whole-model
    /// `recommended_mode`). `Some(_)` for v2.0.0+ when a
    /// populated policy block is present.
    pub per_tensor_policy: Option<PerTensorPolicy>,
    /// Path the manifest was loaded from. Logged at load time
    /// so an operator running multiple checkpoints can see which
    /// file influenced the mode choice.
    pub source: PathBuf,
}

impl NumcertManifest {
    /// Resolve the precision mode for a specific tensor name.
    ///
    /// Priority:
    ///   1. `per_tensor_policy` (when present, schema v2.0.0+):
    ///      glob overrides first, then `default_mode`.
    ///   2. `recommended_mode` (whole-model fallback when
    ///      `per_tensor_policy` is `None`).
    pub fn resolve_for(&self, tensor_name: &str) -> MatmulMode {
        match &self.per_tensor_policy {
            Some(p) => p.resolve(tensor_name),
            None => self.recommended_mode,
        }
    }
}

/// **M10.3.1.1** — minimal glob matcher. Supports any number of
/// `*` wildcards. Each `*` matches zero or more characters; the
/// fixed segments between them must appear in order without
/// overlap. No-`*` patterns require an exact match.
///
/// This shape matches every FFN-routing pattern in the M4.6
/// fixture (`*.mlp.*_proj.weight`, `*.self_attn.*_proj.weight`,
/// `lm_head.weight`) without pulling in a regex dependency.
/// `?` and character classes are not supported — the schema
/// keeps to literal segments only.
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    // No `*` at all: exact match required.
    if segments.len() == 1 {
        return pattern == name;
    }
    // First segment must be a prefix of `name`.
    let first = segments[0];
    if !name.starts_with(first) {
        return false;
    }
    // Last segment must be a suffix of `name`.
    let last = segments[segments.len() - 1];
    if !name.ends_with(last) {
        return false;
    }
    // The total fixed length must fit inside `name`.
    let fixed_len: usize = segments.iter().map(|s| s.len()).sum();
    if name.len() < fixed_len {
        return false;
    }
    // Walk the middle segments in order, advancing the cursor
    // past the prefix and stopping before the suffix.
    let mut cursor = first.len();
    let suffix_start = name.len() - last.len();
    for seg in &segments[1..segments.len() - 1] {
        if seg.is_empty() {
            // Adjacent `**` — equivalent to a single `*`. Skip.
            continue;
        }
        match name[cursor..suffix_start].find(seg) {
            Some(pos) => cursor += pos + seg.len(),
            None => return false,
        }
    }
    true
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
    let text = std::str::from_utf8(bytes).map_err(|e| format!("not valid UTF-8: {e}"))?;

    let schema_version = extract_string_field(text, "schema_version")
        .ok_or_else(|| "missing or non-string \"schema_version\"".to_string())?;
    let recommended_mode_str = extract_string_field(text, "recommended_mode")
        .ok_or_else(|| "missing or non-string \"recommended_mode\"".to_string())?;
    let recommended_mode =
        parse_mode(&recommended_mode_str).map_err(|e| format!("recommended_mode: {e}"))?;
    let per_tensor_policy = parse_per_tensor_policy(text)?;
    Ok(NumcertManifest {
        schema_version,
        recommended_mode,
        per_tensor_policy,
        source,
    })
}

/// Convert the manifest's mode string to the `MatmulMode` enum.
fn parse_mode(value: &str) -> Result<MatmulMode, String> {
    match value {
        "certified" => Ok(MatmulMode::Certified),
        "fast" => Ok(MatmulMode::Fast),
        "quantized" => Ok(MatmulMode::Quantized),
        other => Err(format!(
            "unrecognised mode \"{other}\" (expected \"certified\", \"fast\", or \"quantized\")"
        )),
    }
}

/// **M10.3.1.1** — extract the optional `per_tensor_policy`
/// block. Returns `Ok(None)` for v1.0.0 manifests where the
/// field is `null` or absent, `Ok(Some(_))` for v2.0.0
/// manifests with a populated policy, and `Err` for malformed
/// blocks.
///
/// Pure-bytes scanning to keep the manifest dependency-free; the
/// block we parse has the shape:
///
/// ```text
/// "per_tensor_policy": {
///   "default": "certified",
///   "overrides": [
///     {"pattern": "*.mlp.*_proj.weight", "mode": "fast"}
///   ]
/// }
/// ```
fn parse_per_tensor_policy(text: &str) -> Result<Option<PerTensorPolicy>, String> {
    let key = "\"per_tensor_policy\"";
    let key_pos = match text.find(key) {
        Some(p) => p,
        None => return Ok(None),
    };
    let after_key = &text[key_pos + key.len()..];
    let colon = after_key
        .find(':')
        .ok_or_else(|| "per_tensor_policy missing colon".to_string())?;
    let after_colon = after_key[colon + 1..].trim_start();

    // v1.0.0 path: "per_tensor_policy": null
    if after_colon.starts_with("null") {
        return Ok(None);
    }
    if !after_colon.starts_with('{') {
        return Err("per_tensor_policy must be either null or an object".to_string());
    }

    // v2.0.0+ path: an object with `default` and `overrides`.
    // Find the matching closing brace by counting nesting.
    let block_end = find_balanced_close(after_colon, '{', '}')
        .ok_or_else(|| "per_tensor_policy block has unbalanced braces".to_string())?;
    let block = &after_colon[..=block_end];

    let default_str = extract_string_field(block, "default")
        .ok_or_else(|| "per_tensor_policy missing \"default\"".to_string())?;
    let default_mode =
        parse_mode(&default_str).map_err(|e| format!("per_tensor_policy default: {e}"))?;

    // Optional overrides array. Absent = empty Vec.
    let overrides = parse_overrides_array(block)?;

    Ok(Some(PerTensorPolicy {
        default_mode,
        overrides,
    }))
}

/// Find the position of the closing delimiter that balances the
/// opening delimiter at index 0 of `text`. Counts nesting; ignores
/// delimiters inside string literals (single quote-pair scope).
fn find_balanced_close(text: &str, open: char, close: char) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || bytes[0] as char != open {
        return None;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, &b) in bytes.iter().enumerate() {
        let c = b as char;
        if escape_next {
            escape_next = false;
            continue;
        }
        if in_string {
            if c == '\\' {
                escape_next = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
        } else if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Parse the `overrides` array inside the per-tensor policy
/// block. Each entry is `{"pattern": "...", "mode": "..."}`.
/// Returns an empty Vec if the array is absent.
fn parse_overrides_array(block: &str) -> Result<Vec<PolicyOverride>, String> {
    let key = "\"overrides\"";
    let key_pos = match block.find(key) {
        Some(p) => p,
        None => return Ok(Vec::new()),
    };
    let after_key = &block[key_pos + key.len()..];
    let colon = after_key
        .find(':')
        .ok_or_else(|| "overrides missing colon".to_string())?;
    let after_colon = after_key[colon + 1..].trim_start();
    if !after_colon.starts_with('[') {
        return Err("overrides must be an array".to_string());
    }
    let array_end = find_balanced_close(after_colon, '[', ']')
        .ok_or_else(|| "overrides array has unbalanced brackets".to_string())?;
    let array_body = &after_colon[1..array_end];

    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < array_body.len() {
        let rest = &array_body[cursor..];
        let entry_start = match rest.find('{') {
            Some(p) => p,
            None => break, // trailing whitespace / commas only
        };
        let entry_slice = &rest[entry_start..];
        let entry_end = find_balanced_close(entry_slice, '{', '}')
            .ok_or_else(|| "overrides entry has unbalanced braces".to_string())?;
        let entry = &entry_slice[..=entry_end];

        let pattern = extract_string_field(entry, "pattern")
            .ok_or_else(|| "overrides entry missing \"pattern\"".to_string())?;
        let mode_str = extract_string_field(entry, "mode")
            .ok_or_else(|| "overrides entry missing \"mode\"".to_string())?;
        let mode = parse_mode(&mode_str).map_err(|e| format!("overrides entry: {e}"))?;
        out.push(PolicyOverride { pattern, mode });

        cursor += entry_start + entry_end + 1;
    }
    Ok(out)
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
        let m = parse_manifest(json, fake_source()).expect("manifest should parse");
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
        let m = parse_manifest(json, fake_source()).expect("manifest should parse");
        assert_eq!(m.recommended_mode, MatmulMode::Certified);
    }

    #[test]
    fn parse_manifest_recognises_recommended_mode_quantized() {
        let json = br#"{
            "schema_version": "2.0.0",
            "schema_variant": "gguf-functional",
            "recommended_mode": "quantized",
            "per_tensor_policy": {
                "default": "quantized",
                "overrides": []
            }
        }"#;
        let m = parse_manifest(json, fake_source()).expect("quantized GGUF manifest should parse");
        assert_eq!(m.recommended_mode, MatmulMode::Quantized);
        let p = m.per_tensor_policy.as_ref().expect("policy populated");
        assert_eq!(p.default_mode, MatmulMode::Quantized);
        assert_eq!(
            m.resolve_for("model.layers.0.self_attn.q_proj.weight"),
            MatmulMode::Quantized
        );
    }

    #[test]
    fn parse_manifest_rejects_unknown_mode() {
        let json = br#"{
            "schema_version": "1.0.0",
            "recommended_mode": "turbo"
        }"#;
        let err = parse_manifest(json, fake_source()).expect_err("unknown mode should fail");
        assert!(
            err.contains("turbo"),
            "error should mention bad value, got: {err}"
        );
    }

    #[test]
    fn parse_manifest_rejects_missing_recommended_mode() {
        let json = br#"{
            "schema_version": "1.0.0"
        }"#;
        let err = parse_manifest(json, fake_source()).expect_err("missing field should fail");
        assert!(
            err.contains("recommended_mode"),
            "error should name the missing field"
        );
    }

    #[test]
    fn extract_string_field_handles_whitespace() {
        let text = r#"{ "key" :    "value", "other": "x" }"#;
        assert_eq!(extract_string_field(text, "key"), Some("value".to_string()));
        assert_eq!(extract_string_field(text, "other"), Some("x".to_string()));
        assert_eq!(extract_string_field(text, "missing"), None);
    }

    // ----- M10.3.1.1 — glob matcher -----

    #[test]
    fn glob_match_exact_no_wildcard() {
        assert!(glob_match("lm_head.weight", "lm_head.weight"));
        assert!(!glob_match("lm_head.weight", "lm_head.bias"));
    }

    #[test]
    fn glob_match_prefix_pattern() {
        assert!(glob_match(
            "model.layers.*",
            "model.layers.0.mlp.gate_proj.weight"
        ));
        assert!(!glob_match("model.layers.*", "embed_tokens.weight"));
    }

    #[test]
    fn glob_match_suffix_pattern() {
        assert!(glob_match(
            "*.weight",
            "model.layers.0.mlp.gate_proj.weight"
        ));
        assert!(!glob_match("*.weight", "model.layers.0.mlp.gate_proj.bias"));
    }

    #[test]
    fn glob_match_contains_pattern() {
        assert!(glob_match(
            "*.mlp.*_proj.weight",
            "model.layers.0.mlp.gate_proj.weight"
        ));
        assert!(glob_match(
            "*.mlp.*_proj.weight",
            "model.layers.31.mlp.down_proj.weight"
        ));
        assert!(!glob_match(
            "*.mlp.*_proj.weight",
            "model.layers.0.self_attn.q_proj.weight"
        ));
    }

    #[test]
    fn glob_match_multiwildcard_in_order() {
        // Multi-* patterns: each segment must appear in order.
        assert!(glob_match("*foo*bar*", "abc.foo.xyz.bar.qed"));
        // Out-of-order segments fail the match.
        assert!(!glob_match("*foo*bar*", "abc.bar.xyz.foo.qed"));
    }

    // ----- M10.3.1.1 — per-tensor policy parsing -----

    #[test]
    fn parse_v1_manifest_with_null_policy() {
        let json = br#"{
            "schema_version": "1.0.0",
            "recommended_mode": "certified",
            "per_tensor_policy": null
        }"#;
        let m =
            parse_manifest(json, fake_source()).expect("v1 manifest with null policy should parse");
        assert!(m.per_tensor_policy.is_none());
        assert_eq!(m.recommended_mode, MatmulMode::Certified);
        // resolve_for falls back to recommended_mode for any name
        assert_eq!(
            m.resolve_for("model.layers.0.mlp.gate_proj.weight"),
            MatmulMode::Certified
        );
    }

    #[test]
    fn parse_v2_manifest_with_overrides() {
        let json = br#"{
            "schema_version": "2.0.0",
            "recommended_mode": "certified",
            "per_tensor_policy": {
                "default": "certified",
                "overrides": [
                    {"pattern": "*.mlp.*_proj.weight", "mode": "fast"}
                ]
            }
        }"#;
        let m = parse_manifest(json, fake_source()).expect("v2 manifest should parse");
        let p = m.per_tensor_policy.as_ref().expect("policy populated");
        assert_eq!(p.default_mode, MatmulMode::Certified);
        assert_eq!(p.overrides.len(), 1);
        assert_eq!(p.overrides[0].pattern, "*.mlp.*_proj.weight");
        assert_eq!(p.overrides[0].mode, MatmulMode::Fast);
        // FFN tensors take fast; attention tensors fall through
        // to default certified.
        assert_eq!(
            m.resolve_for("model.layers.0.mlp.gate_proj.weight"),
            MatmulMode::Fast
        );
        assert_eq!(
            m.resolve_for("model.layers.0.self_attn.q_proj.weight"),
            MatmulMode::Certified
        );
    }

    #[test]
    fn parse_v2_manifest_without_overrides_array() {
        let json = br#"{
            "schema_version": "2.0.0",
            "recommended_mode": "fast",
            "per_tensor_policy": {
                "default": "fast"
            }
        }"#;
        let m = parse_manifest(json, fake_source())
            .expect("v2 manifest with bare default should parse");
        let p = m.per_tensor_policy.as_ref().expect("policy populated");
        assert_eq!(p.default_mode, MatmulMode::Fast);
        assert!(p.overrides.is_empty());
        assert_eq!(m.resolve_for("anything.weight"), MatmulMode::Fast);
    }

    #[test]
    fn parse_v2_manifest_first_match_wins() {
        let json = br#"{
            "schema_version": "2.0.0",
            "recommended_mode": "certified",
            "per_tensor_policy": {
                "default": "certified",
                "overrides": [
                    {"pattern": "*.mlp.down_proj.weight", "mode": "certified"},
                    {"pattern": "*.mlp.*_proj.weight", "mode": "fast"}
                ]
            }
        }"#;
        let m = parse_manifest(json, fake_source()).expect("v2 manifest should parse");
        // down_proj matches the first (more-specific) override
        // before the broader one.
        assert_eq!(
            m.resolve_for("model.layers.0.mlp.down_proj.weight"),
            MatmulMode::Certified
        );
        // gate_proj only matches the second override.
        assert_eq!(
            m.resolve_for("model.layers.0.mlp.gate_proj.weight"),
            MatmulMode::Fast
        );
    }

    #[test]
    fn parse_manifest_rejects_malformed_per_tensor_policy() {
        let json = br#"{
            "schema_version": "2.0.0",
            "recommended_mode": "certified",
            "per_tensor_policy": "not an object or null"
        }"#;
        let err = parse_manifest(json, fake_source()).expect_err("string policy should fail");
        assert!(
            err.contains("per_tensor_policy"),
            "error should name field, got: {err}"
        );
    }
}
