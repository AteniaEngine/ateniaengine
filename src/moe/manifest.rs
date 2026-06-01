//! **MOE-FULL-14** — machine-readable MoE certification manifest.
//!
//! The controlled production path ([`super::production`]) consults this manifest
//! to decide whether a detected MoE family may run. It is the single source of
//! truth for "what is certified", embedded into the binary from the committed
//! `fixtures/moe/moe_cert_manifest.json` (so the gating never depends on an
//! external file). It is **tiny-fixture** certification, **not** a large-model
//! production certification (kept deliberately separate from the dense
//! `nn::llama::numcert` matmul-precision manifest).

use super::family::MoeFamily;

/// Certification scope for a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeCertScope {
    /// Numerically certified vs HF f64 on a tiny **end-to-end** fixture.
    CertifiedFixture,
    /// Also has a certified **real-checkpoint** single-layer MoE-block
    /// sub-reference (stronger evidence than a purely synthetic fixture).
    CertifiedPartial,
    /// Also certified on a **topology-representative scale** fixture that mirrors
    /// the real large-checkpoint structure (expert count, top-k, GQA, shared,
    /// MLA) — NOT the multi-GB real weights (MOE-FULL-15).
    CertifiedScaled,
    /// Runnable but not certified.
    Experimental,
    /// Recognised but not supported (e.g. an attention variant not implemented).
    Unsupported,
}

impl MoeCertScope {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "certified_fixture" => Some(Self::CertifiedFixture),
            "certified_partial" => Some(Self::CertifiedPartial),
            "certified_scaled" => Some(Self::CertifiedScaled),
            "experimental" => Some(Self::Experimental),
            "unsupported" => Some(Self::Unsupported),
            _ => None,
        }
    }

    /// Whether this scope is allowed to run on the controlled production path.
    pub fn is_runnable(self) -> bool {
        matches!(self, Self::CertifiedFixture | Self::CertifiedPartial | Self::CertifiedScaled)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CertifiedFixture => "certified_fixture",
            Self::CertifiedPartial => "certified_partial",
            Self::CertifiedScaled => "certified_scaled",
            Self::Experimental => "experimental",
            Self::Unsupported => "unsupported",
        }
    }
}

/// One family's certification record (the fields the gating + diagnostics need).
#[derive(Debug, Clone)]
pub struct MoeCertEntry {
    pub family: MoeFamily,
    pub scope: MoeCertScope,
    pub runtime_path: String,
    pub attention: String,
    pub moe_layout: String,
    pub end_to_end_drift_max_abs: f64,
    pub argmax_match: bool,
    pub generate_eos: bool,
    pub limitations: String,
}

/// An unsupported-variant fingerprint (a tensor-name marker → reason).
#[derive(Debug, Clone)]
pub struct UnsupportedVariant {
    pub variant: String,
    pub marker: String,
    pub reason: String,
}

/// The parsed manifest.
#[derive(Debug, Clone)]
pub struct MoeCertManifest {
    pub version: u64,
    pub families: Vec<MoeCertEntry>,
    pub unsupported_variants: Vec<UnsupportedVariant>,
}

const EMBEDDED: &str = include_str!("../../fixtures/moe/moe_cert_manifest.json");

fn family_from_str(s: &str) -> Option<MoeFamily> {
    match s {
        "Mixtral" => Some(MoeFamily::Mixtral),
        "Qwen-MoE" => Some(MoeFamily::QwenMoe),
        "DeepSeek-MoE" => Some(MoeFamily::DeepSeekMoe),
        _ => None,
    }
}

impl MoeCertManifest {
    /// The built-in manifest (embedded at compile time). Panics only if the
    /// committed JSON is malformed — caught by `manifest_parses` in tests.
    pub fn builtin() -> Self {
        Self::parse(EMBEDDED).expect("embedded moe_cert_manifest.json must parse")
    }

    /// Parse a manifest JSON string.
    pub fn parse(json: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("manifest json: {e}"))?;
        let version = v.get("version").and_then(|x| x.as_u64()).unwrap_or(0);
        let mut families = Vec::new();
        for f in v.get("families").and_then(|x| x.as_array()).into_iter().flatten() {
            let fam = f
                .get("family")
                .and_then(|x| x.as_str())
                .and_then(family_from_str)
                .ok_or_else(|| "manifest: unknown/missing family".to_string())?;
            let scope = f
                .get("scope")
                .and_then(|x| x.as_str())
                .and_then(MoeCertScope::parse)
                .ok_or_else(|| "manifest: unknown/missing scope".to_string())?;
            families.push(MoeCertEntry {
                family: fam,
                scope,
                runtime_path: f.get("runtime_path").and_then(|x| x.as_str()).unwrap_or("").into(),
                attention: f.get("attention").and_then(|x| x.as_str()).unwrap_or("").into(),
                moe_layout: f.get("moe_layout").and_then(|x| x.as_str()).unwrap_or("").into(),
                end_to_end_drift_max_abs: f
                    .get("end_to_end_drift_max_abs")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(f64::NAN),
                argmax_match: f.get("argmax_match").and_then(|x| x.as_bool()).unwrap_or(false),
                generate_eos: f.get("generate_eos").and_then(|x| x.as_bool()).unwrap_or(false),
                limitations: f.get("limitations").and_then(|x| x.as_str()).unwrap_or("").into(),
            });
        }
        let mut unsupported_variants = Vec::new();
        for u in v.get("unsupported_variants").and_then(|x| x.as_array()).into_iter().flatten() {
            unsupported_variants.push(UnsupportedVariant {
                variant: u.get("variant").and_then(|x| x.as_str()).unwrap_or("").into(),
                marker: u.get("marker").and_then(|x| x.as_str()).unwrap_or("").into(),
                reason: u.get("reason").and_then(|x| x.as_str()).unwrap_or("").into(),
            });
        }
        Ok(Self { version, families, unsupported_variants })
    }

    /// The certification entry for a family, if present.
    pub fn entry(&self, family: MoeFamily) -> Option<&MoeCertEntry> {
        self.families.iter().find(|e| e.family == family)
    }

    /// The certification scope for a family (`Experimental` if absent — a
    /// recognised family with no manifest entry is not runnable).
    pub fn scope_for(&self, family: MoeFamily) -> MoeCertScope {
        self.entry(family).map(|e| e.scope).unwrap_or(MoeCertScope::Experimental)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parses_and_lists_three_families() {
        let m = MoeCertManifest::builtin();
        assert_eq!(m.version, 2);
        assert_eq!(m.families.len(), 3);
        assert!(m.scope_for(MoeFamily::Mixtral).is_runnable());
        assert!(m.scope_for(MoeFamily::QwenMoe).is_runnable());
        assert!(m.scope_for(MoeFamily::DeepSeekMoe).is_runnable());
        // MOE-FULL-15: all three reached topology-representative scale cert.
        assert_eq!(m.scope_for(MoeFamily::Mixtral), MoeCertScope::CertifiedScaled);
        assert_eq!(m.scope_for(MoeFamily::QwenMoe), MoeCertScope::CertifiedScaled);
        assert_eq!(m.scope_for(MoeFamily::DeepSeekMoe), MoeCertScope::CertifiedScaled);
        assert!(m.unsupported_variants.iter().any(|u| u.variant.contains("Qwen3")));
    }

    #[test]
    fn scopes_runnable_semantics() {
        assert!(MoeCertScope::CertifiedFixture.is_runnable());
        assert!(MoeCertScope::CertifiedPartial.is_runnable());
        assert!(!MoeCertScope::Experimental.is_runnable());
        assert!(!MoeCertScope::Unsupported.is_runnable());
    }
}
