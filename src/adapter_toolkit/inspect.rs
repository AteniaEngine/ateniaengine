//! **Adapter Toolkit v2 — Parts 8/9: model inspection & auto-detection.**
//!
//! [`inspect_model_dir`] reads a model directory and produces an
//! [`AdapterDsl`] describing it — the engine behind `atenia
//! inspect`. It detects:
//! - **format** — a `*.gguf` file ⇒ GGUF, a `config.json` ⇒ HF
//!   safetensors;
//! - **family / architecture** — from `architectures[0]` /
//!   `model_type` (HF) or `general.architecture` (GGUF);
//! - **attention shape** — MHA / GQA / MQA from the head counts;
//! - **EOS set** — `eos_token_id` (HF) or `tokenizer.ggml.eos_token_id`
//!   (GGUF);
//! - **RoPE variant** — `longrope` / `partial` / `standard` from
//!   the `rope_scaling` block. **HF only.** GGUF metadata does not
//!   expose the RoPE variant (llama.cpp folds it into the
//!   `rope_factors` tensors), so a GGUF is always reported as
//!   `standard` *plus an explicit note* in [`InspectionReport::notes`]
//!   warning that a long-context model may need `config.rope`
//!   added by hand. Auto-detection never invents metadata.
//!
//! The emitted [`AdapterDsl`] is guaranteed to round-trip: it
//! resolves through [`ResolvedAdapterSpec`] and passes
//! [`super::validate::validate`], so `atenia inspect` output can be
//! fed straight back into `atenia load`. Inspection is read-only
//! and never builds a graph.

use std::path::Path;

use crate::v17::loader::gguf_reader::GgufReader;

use super::dsl::{
    AdapterDsl, AttentionSection, ConfigSection, KvHeads, TokenizerSection,
};
use super::ToolkitError;

/// The result of inspecting a model directory: the synthesised
/// [`AdapterDsl`] plus any non-fatal detection notes (e.g. a
/// feature that the source format does not expose and therefore
/// could not be auto-detected).
#[derive(Debug, Clone, PartialEq)]
pub struct InspectionReport {
    /// The auto-detected DSL — guaranteed to validate and resolve.
    pub dsl: AdapterDsl,
    /// Detection notes the caller should surface to the user.
    /// These are limitations of auto-detection, not errors.
    pub notes: Vec<String>,
}

/// Inspect a model directory and synthesise an [`InspectionReport`].
pub fn inspect_model_dir(dir: &Path) -> Result<InspectionReport, ToolkitError> {
    if !dir.is_dir() {
        return Err(ToolkitError::Io(format!(
            "{} is not a directory",
            dir.display()
        )));
    }

    if let Some(gguf) = find_gguf(dir)? {
        inspect_gguf(&gguf)
    } else if dir.join("config.json").is_file() {
        inspect_hf_config(&dir.join("config.json"))
    } else {
        Err(ToolkitError::Io(format!(
            "{} contains neither a config.json nor a .gguf file",
            dir.display()
        )))
    }
}

/// First `*.gguf` file in `dir`, if any.
fn find_gguf(dir: &Path) -> Result<Option<std::path::PathBuf>, ToolkitError> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| ToolkitError::Io(format!("{}: {e}", dir.display())))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

// ----------------------------------------------------------------
// HF safetensors path (config.json)
// ----------------------------------------------------------------

fn inspect_hf_config(config_path: &Path) -> Result<InspectionReport, ToolkitError> {
    let text = std::fs::read_to_string(config_path)
        .map_err(|e| ToolkitError::Io(format!("{}: {e}", config_path.display())))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| ToolkitError::Parse(format!("config.json: {e}")))?;

    let architecture = v
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
        .map(str::to_string);
    let model_type = v.get("model_type").and_then(|s| s.as_str());

    let family = family_from_hf(architecture.as_deref(), model_type)?;

    let n_heads = v
        .get("num_attention_heads")
        .and_then(serde_json::Value::as_u64);
    let n_kv = v
        .get("num_key_value_heads")
        .and_then(serde_json::Value::as_u64)
        .or(n_heads); // absent ⇒ MHA
    let attention = attention_section(n_heads, n_kv);

    let config = rope_section_from_hf(&v);
    let tokenizer = eos_section(eos_ids_from_hf(&v));

    // The HF `config.json` fully specifies the RoPE variant
    // (`rope_scaling` / `partial_rotary_factor`), so safetensors
    // auto-detection is lossless — no notes are produced here.
    Ok(InspectionReport {
        dsl: AdapterDsl {
            family: family.to_string(),
            architecture,
            model_type: model_type.map(str::to_string),
            format: Some("safetensors".to_string()),
            quant: None,
            config,
            weights: None,
            attention,
            tokenizer,
            overrides: Default::default(),
        },
        notes: Vec::new(),
    })
}

fn eos_ids_from_hf(v: &serde_json::Value) -> Vec<u32> {
    match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            n.as_u64().map(|x| vec![x as u32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|e| e.as_u64().map(|x| x as u32))
            .collect(),
        _ => Vec::new(),
    }
}

fn rope_section_from_hf(v: &serde_json::Value) -> Option<ConfigSection> {
    let mut section = ConfigSection::default();
    let mut any = false;

    // `rope_scaling.rope_type` / `.type` — `longrope` is the only
    // variant the toolkit names distinctly today.
    if let Some(scaling) = v.get("rope_scaling").filter(|s| !s.is_null()) {
        let rope_type = scaling
            .get("rope_type")
            .or_else(|| scaling.get("type"))
            .and_then(|s| s.as_str());
        if let Some(rt) = rope_type {
            if rt.eq_ignore_ascii_case("longrope") {
                section.rope = Some("longrope".to_string());
                any = true;
            }
        }
    }

    if let Some(prf) = v
        .get("partial_rotary_factor")
        .and_then(serde_json::Value::as_f64)
    {
        section.partial_rotary_factor = Some(prf as f32);
        if section.rope.is_none() {
            section.rope = Some("partial".to_string());
        }
        any = true;
    }

    if let Some(theta) = v.get("rope_theta").and_then(serde_json::Value::as_f64) {
        section.rope_theta = Some(theta as f32);
        any = true;
    }

    any.then_some(section)
}

// ----------------------------------------------------------------
// GGUF path
// ----------------------------------------------------------------

fn inspect_gguf(gguf_path: &Path) -> Result<InspectionReport, ToolkitError> {
    let reader = GgufReader::read_from_path(gguf_path)
        .map_err(|e| ToolkitError::Io(format!("{}: {e}", gguf_path.display())))?;

    let arch = reader
        .architecture()
        .ok_or_else(|| ToolkitError::Parse("GGUF: missing general.architecture".to_string()))?
        .to_string();
    let family = family_from_gguf(&arch)?;

    let n_heads = meta_u64(&reader, &format!("{arch}.attention.head_count"));
    let n_kv = meta_u64(&reader, &format!("{arch}.attention.head_count_kv")).or(n_heads);
    let attention = attention_section(n_heads, n_kv);

    let eos = meta_u64(&reader, "tokenizer.ggml.eos_token_id")
        .map(|x| vec![x as u32])
        .unwrap_or_default();
    let eos_was_absent = eos.is_empty();
    let tokenizer = eos_section(eos);

    // GGUF metadata carries `rope.freq_base` but NOT a RoPE-variant
    // tag — llama.cpp folds LongRoPE / partial-rotary scaling into
    // the precomputed `rope_factors` tensors and does not surface
    // the variant as a scalar key. Auto-detection therefore cannot
    // tell a plain-RoPE GGUF from a LongRoPE GGUF. We leave `rope`
    // unset (the loader's standard path, correct for every family
    // whose GGUF the engine validates today) and emit an explicit
    // note rather than inventing metadata.
    let config = meta_f32(&reader, &format!("{arch}.rope.freq_base")).map(|theta| ConfigSection {
        rope: None,
        partial_rotary_factor: None,
        rope_theta: Some(theta),
    });

    let mut notes = Vec::new();
    notes.push(
        "GGUF metadata does not expose the RoPE variant; `rope` was left unset \
         (standard). Long-context models (e.g. Phi-3 LongRoPE) may need \
         `config.rope: longrope` added to the YAML by hand — auto-detection \
         cannot recover this from a GGUF file."
            .to_string(),
    );
    if eos_was_absent {
        notes.push(
            "GGUF metadata had no `tokenizer.ggml.eos_token_id`; `tokenizer.eos_tokens` \
             was left unset (the loader will read EOS from the GGUF tokenizer)."
                .to_string(),
        );
    }

    Ok(InspectionReport {
        dsl: AdapterDsl {
            family: family.to_string(),
            // GGUF carries `general.architecture` (e.g. "llama"),
            // not the HF class name; leave `architecture` unset so
            // the family default applies on reload.
            architecture: None,
            model_type: None,
            format: Some("gguf".to_string()),
            quant: None,
            config,
            weights: None,
            attention,
            tokenizer,
            overrides: Default::default(),
        },
        notes,
    })
}

fn meta_u64(reader: &GgufReader, key: &str) -> Option<u64> {
    reader.metadata.get(key).and_then(|v| v.as_u64())
}

fn meta_f32(reader: &GgufReader, key: &str) -> Option<f32> {
    reader.metadata.get(key).and_then(|v| v.as_f32())
}

// ----------------------------------------------------------------
// Shared detection helpers
// ----------------------------------------------------------------

/// Build the `attention:` section from the head counts. `None`
/// counts ⇒ no section emitted (the loader falls back to
/// `config.json`).
fn attention_section(n_heads: Option<u64>, n_kv: Option<u64>) -> Option<AttentionSection> {
    let (h, kv) = (n_heads?, n_kv?);
    let (kind, kv_heads) = if kv == h {
        ("mha", None)
    } else if kv == 1 {
        ("mqa", Some(KvHeads::Count(1)))
    } else {
        ("gqa", Some(KvHeads::Count(kv as usize)))
    };
    Some(AttentionSection {
        kind: Some(kind.to_string()),
        kv_heads,
    })
}

fn eos_section(eos: Vec<u32>) -> Option<TokenizerSection> {
    if eos.is_empty() {
        None
    } else {
        Some(TokenizerSection {
            eos_tokens: Some(eos),
            turn_terminators: None,
        })
    }
}

/// Map an HF `architectures[0]` / `model_type` to a DSL family
/// string. Falls back to `model_type` when the architecture class
/// is unknown.
fn family_from_hf(
    architecture: Option<&str>,
    model_type: Option<&str>,
) -> Result<&'static str, ToolkitError> {
    if let Some(arch) = architecture {
        match arch {
            "LlamaForCausalLM" => return Ok("llama"),
            "Qwen2ForCausalLM" => return Ok("qwen2"),
            "Qwen3ForCausalLM" => return Ok("qwen3"),
            "Gemma2ForCausalLM" => return Ok("gemma2"),
            "Gemma3ForCausalLM" | "Gemma3ForConditionalGeneration" => return Ok("gemma3"),
            "Phi3ForCausalLM" => return Ok("phi3"),
            "MistralForCausalLM" => return Ok("mistral"),
            _ => {}
        }
    }
    match model_type {
        Some("llama") => Ok("llama"),
        Some("qwen2") => Ok("qwen2"),
        Some("qwen3") => Ok("qwen3"),
        Some("gemma2") => Ok("gemma2"),
        Some("gemma3" | "gemma3_text") => Ok("gemma3"),
        Some("phi3") => Ok("phi3"),
        Some("mistral") => Ok("mistral"),
        _ => Err(ToolkitError::Resolution(format!(
            "cannot auto-detect a supported family from architecture={architecture:?} \
             model_type={model_type:?} — Adapter Toolkit v2 supports llama, qwen2, qwen3, \
             gemma2, gemma3, phi3, mistral"
        ))),
    }
}

/// Map a GGUF `general.architecture` to a DSL family string.
fn family_from_gguf(arch: &str) -> Result<&'static str, ToolkitError> {
    match arch {
        "llama" => Ok("llama"),
        "qwen2" => Ok("qwen2"),
        "qwen3" => Ok("qwen3"),
        "gemma2" => Ok("gemma2"),
        "gemma3" => Ok("gemma3"),
        "phi3" => Ok("phi3"),
        other => Err(ToolkitError::Resolution(format!(
            "GGUF general.architecture = `{other}` has no Adapter Toolkit v2 family \
             (supported: llama, qwen2, qwen3, gemma2, gemma3, phi3)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_toolkit::spec::ResolvedAdapterSpec;
    use crate::adapter_toolkit::validate::validate;

    fn write_config(dir: &Path, json: &str) {
        std::fs::write(dir.join("config.json"), json).expect("write config");
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("atk_inspect_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("mkdir");
        d
    }

    #[test]
    fn inspects_llama_gqa_config() {
        let dir = tmp_dir("llama");
        write_config(
            &dir,
            r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama",
                "num_attention_heads":32,"num_key_value_heads":8,"eos_token_id":2}"#,
        );
        let dsl = inspect_model_dir(&dir).expect("inspects").dsl;
        assert_eq!(dsl.family, "llama");
        assert_eq!(dsl.format.as_deref(), Some("safetensors"));
        let att = dsl.attention.as_ref().unwrap();
        assert_eq!(att.kind.as_deref(), Some("gqa"));
        assert_eq!(att.kv_heads, Some(KvHeads::Count(8)));
        assert_eq!(
            dsl.tokenizer.as_ref().unwrap().eos_tokens.as_deref(),
            Some([2].as_slice())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inspected_dsl_round_trips_through_load_path() {
        let dir = tmp_dir("rt");
        write_config(
            &dir,
            r#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2",
                "num_attention_heads":28,"num_key_value_heads":4,
                "eos_token_id":[151643,151645]}"#,
        );
        let dsl = inspect_model_dir(&dir).expect("inspects").dsl;
        // Must validate and resolve — the `atenia load` contract.
        assert!(validate(&dsl).is_ok(), "inspected DSL must validate");
        let spec = ResolvedAdapterSpec::resolve(&dsl).expect("inspected DSL must resolve");
        assert_eq!(spec.architecture, "Qwen2ForCausalLM");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detects_mha_when_kv_equals_heads() {
        let dir = tmp_dir("mha");
        write_config(
            &dir,
            r#"{"architectures":["LlamaForCausalLM"],"num_attention_heads":32,
                "num_key_value_heads":32,"eos_token_id":2}"#,
        );
        let dsl = inspect_model_dir(&dir).expect("inspects").dsl;
        assert_eq!(dsl.attention.unwrap().kind.as_deref(), Some("mha"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detects_longrope_from_rope_scaling() {
        let dir = tmp_dir("longrope");
        write_config(
            &dir,
            r#"{"architectures":["Phi3ForCausalLM"],"model_type":"phi3",
                "num_attention_heads":32,"num_key_value_heads":32,
                "rope_scaling":{"rope_type":"longrope"},"eos_token_id":32000}"#,
        );
        let dsl = inspect_model_dir(&dir).expect("inspects").dsl;
        assert_eq!(
            dsl.config.as_ref().unwrap().rope.as_deref(),
            Some("longrope")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_architecture_fails_loud() {
        let dir = tmp_dir("falcon");
        write_config(
            &dir,
            r#"{"architectures":["FalconForCausalLM"],"model_type":"falcon"}"#,
        );
        let err = inspect_model_dir(&dir);
        assert!(matches!(err, Err(ToolkitError::Resolution(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_dir_is_typed_error() {
        let dir = tmp_dir("empty");
        let err = inspect_model_dir(&dir);
        assert!(matches!(err, Err(ToolkitError::Io(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
