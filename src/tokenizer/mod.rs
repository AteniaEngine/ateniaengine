//! M5.a — public tokenizer surface for Atenia Engine.
//!
//! `AteniaTokenizer` wraps the HuggingFace `tokenizers` crate
//! (pure Rust, byte-exact with the Python `transformers`
//! reference implementation per R1) and adds Jinja2-rendered
//! chat-template support via `minijinja` so the same surface
//! handles every model family in the M4.6 / M4.7 validated set
//! (Llama 2, Llama 3.2, Qwen 2.5, SmolLM2, TinyLlama). All
//! template rendering reads from each checkpoint's own
//! `tokenizer_config.json` — no per-family branching, per
//! decision D65.
//!
//! ## What lives where
//!
//! | File | Source of truth | Field consumed |
//! |---|---|---|
//! | `tokenizer.json` | HF tokenizer state | full vocab + merges + pre/post processors |
//! | `tokenizer_config.json` | HF runtime config | `chat_template`, `bos_token`, `eos_token`, `add_bos_token`, `add_eos_token` |
//! | `special_tokens_map.json` | redundant w/ above | not consumed (covered by config) |
//!
//! ## Public API surface
//!
//! ```ignore
//! let tok = AteniaTokenizer::from_model_dir("models/llama-2-13b-chat")?;
//! let ids = tok.encode("Hello, my name is", true);          // BOS prepended
//! let txt = tok.decode(&ids, true);                          // skip specials
//! let prompt = tok.apply_chat_template(&[
//!     ChatMessage::user("Hi!"),
//! ])?;                                                       // Jinja2 rendered
//! let bos = tok.bos_id();                                    // 1 for Llama 2
//! let eos = tok.eos_id();                                    // 2 for Llama 2
//! ```
//!
//! The encode/decode contract matches HF byte-exactly; the
//! chat-template contract matches HF on the same Jinja2 engine
//! variant the `transformers` package uses (`minijinja` mirrors
//! the subset HF actually exercises).

use std::path::{Path, PathBuf};
use std::fs;

use serde::Deserialize;
use tokenizers::Tokenizer as HfTokenizer;
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;

/// One message in a chat conversation. Roles map to whatever
/// the underlying chat template expects — typical values are
/// `"system"`, `"user"`, `"assistant"`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

/// Errors surfaced by `AteniaTokenizer`. Kept simple — the
/// expected error rate of this code path is "checkpoint files
/// missing" or "Jinja2 template failed to render", not deep
/// diagnostics.
#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Hf(String),
    Template(String),
    MissingField(&'static str),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Io(e) => write!(f, "tokenizer io: {e}"),
            TokenizerError::Json(e) => write!(f, "tokenizer config json: {e}"),
            TokenizerError::Hf(s) => write!(f, "tokenizers crate: {s}"),
            TokenizerError::Template(s) => write!(f, "chat template: {s}"),
            TokenizerError::MissingField(s) => write!(f, "tokenizer_config.json missing field: {s}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self { TokenizerError::Io(e) }
}
impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self { TokenizerError::Json(e) }
}

/// HF-style "AddedToken" — sometimes a bare string, sometimes
/// an object `{ "content": "<s>", ... }`. We only care about
/// `content`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AddedTokenLike {
    Str(String),
    Obj { content: String },
}

impl AddedTokenLike {
    fn content(self) -> String {
        match self {
            AddedTokenLike::Str(s) => s,
            AddedTokenLike::Obj { content } => content,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct TokenizerConfigRaw {
    #[serde(default)]
    add_bos_token: Option<bool>,
    #[serde(default)]
    add_eos_token: Option<bool>,
    #[serde(default)]
    bos_token: Option<AddedTokenLike>,
    #[serde(default)]
    eos_token: Option<AddedTokenLike>,
    #[serde(default)]
    #[allow(dead_code)] // surface kept for future M5.+ wiring
    pad_token: Option<AddedTokenLike>,
    #[serde(default)]
    #[allow(dead_code)]
    unk_token: Option<AddedTokenLike>,
    #[serde(default)]
    chat_template: Option<ChatTemplateField>,
}

/// `chat_template` in `tokenizer_config.json` is usually a
/// string but Llama 3.x sometimes ships a list of named
/// templates `[{"name": "default", "template": "..."}]`. We
/// pick the entry named `"default"`, falling back to the
/// first.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatTemplateField {
    Single(String),
    Multi(Vec<ChatTemplateEntry>),
}

#[derive(Debug, Deserialize)]
struct ChatTemplateEntry {
    name: String,
    template: String,
}

impl ChatTemplateField {
    fn into_default_template(self) -> Option<String> {
        match self {
            ChatTemplateField::Single(s) => Some(s),
            ChatTemplateField::Multi(entries) => {
                entries
                    .iter()
                    .find(|e| e.name == "default")
                    .map(|e| e.template.clone())
                    .or_else(|| entries.into_iter().next().map(|e| e.template))
            }
        }
    }
}

/// The high-level tokenizer surface used by the rest of
/// Atenia (M5.b will plug it ahead of the prefill graph;
/// M5.d uses it both for prompt encoding and per-token
/// streaming `decode`).
pub struct AteniaTokenizer {
    inner: HfTokenizer,
    /// `None` when the model has no global BOS sentinel
    /// (Qwen 2.5 is the canonical example — conversation
    /// boundaries are marked by `<|im_start|>` instead).
    bos_id: Option<u32>,
    eos_id: u32,
    bos_token: Option<String>,
    eos_token: String,
    add_bos_token: bool,
    add_eos_token: bool,
    chat_template: Option<String>,
    model_dir: PathBuf,
}

impl AteniaTokenizer {
    /// Load a tokenizer from a model directory. Expects at
    /// minimum `tokenizer.json` and `tokenizer_config.json`.
    /// The function is intentionally infallible-on-missing
    /// `chat_template` — `apply_chat_template` will fail later
    /// with a clear error rather than blocking raw-prompt use
    /// cases.
    pub fn from_model_dir<P: AsRef<Path>>(dir: P) -> Result<Self, TokenizerError> {
        let dir = dir.as_ref().to_path_buf();
        let tokenizer_json = dir.join("tokenizer.json");
        let tokenizer_config_json = dir.join("tokenizer_config.json");

        let inner = HfTokenizer::from_file(&tokenizer_json)
            .map_err(|e| TokenizerError::Hf(format!("loading {}: {e}", tokenizer_json.display())))?;

        let cfg_raw = fs::read_to_string(&tokenizer_config_json)?;
        let cfg: TokenizerConfigRaw = serde_json::from_str(&cfg_raw)?;

        // `bos_token` is **optional** — Qwen 2.5 ships a
        // chat-only model whose `tokenizer_config.json` has no
        // global BOS. `eos_token` is universal across the M5
        // model scope.
        let bos_token = cfg.bos_token.map(|t| t.content());
        let eos_token = cfg.eos_token.map(|t| t.content())
            .ok_or(TokenizerError::MissingField("eos_token"))?;

        let bos_id = match &bos_token {
            Some(t) => Some(inner.token_to_id(t)
                .ok_or_else(|| TokenizerError::Hf(format!("bos_token {t:?} not in vocab")))?),
            None => None,
        };
        let eos_id = inner.token_to_id(&eos_token)
            .ok_or_else(|| TokenizerError::Hf(format!("eos_token {eos_token:?} not in vocab")))?;

        // If the config doesn't specify, default to the model's
        // own answer: `add_bos_token` defaults to `bos_token.is_some()`
        // for the chat-fine-tuned families that omit the flag.
        let add_bos_token = cfg.add_bos_token.unwrap_or(bos_token.is_some());
        let add_eos_token = cfg.add_eos_token.unwrap_or(false);

        let chat_template = cfg.chat_template.and_then(|t| t.into_default_template());

        Ok(Self {
            inner,
            bos_id,
            eos_id,
            bos_token,
            eos_token,
            add_bos_token,
            add_eos_token,
            chat_template,
            model_dir: dir,
        })
    }

    /// Encode `text` to token IDs.
    ///
    /// `add_bos = true` honours the model's `add_bos_token`
    /// flag (most chat models prepend `<s>` / `<|begin_of_text|>`
    /// / equivalent). Pass `false` to encode mid-sequence
    /// continuations where BOS would corrupt the context.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, TokenizerError> {
        // `add_special_tokens = false` keeps us in full control of
        // BOS handling; we layer it on ourselves below so the
        // `add_bos` knob is honoured even when `add_bos_token` in
        // the config is `false` (e.g. mid-stream continuations).
        let enc = self.inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Hf(format!("encode: {e}")))?;
        let mut ids = enc.get_ids().to_vec();

        if add_bos && self.add_bos_token {
            if let Some(bos) = self.bos_id {
                ids.insert(0, bos);
            }
        }
        if self.add_eos_token {
            ids.push(self.eos_id);
        }
        Ok(ids)
    }

    /// Decode `ids` back to text.
    ///
    /// `skip_special_tokens = true` strips BOS/EOS/etc. so the
    /// generation loop in M5.d can `print!` decoded chunks
    /// without leaking sentinels into the user-visible output.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Hf(format!("decode: {e}")))
    }

    /// Render the model's `chat_template` (Jinja2) over a
    /// list of messages. Returns the full prompt string ready
    /// to feed to `encode(..., add_bos = true)`.
    ///
    /// Errors with `TokenizerError::MissingField("chat_template")`
    /// if the model's `tokenizer_config.json` had no template;
    /// raw-prompt callers should not hit this path.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String, TokenizerError> {
        let template_src = self.chat_template.as_deref()
            .ok_or(TokenizerError::MissingField("chat_template"))?;

        let mut env = Environment::new();
        // Wire Python-compat string/list methods (`.strip()`,
        // `.startswith()`, `.replace()`, ...). Llama 2's
        // template uses `.strip()` on its very first line; HF
        // templates assume CPython-style methods generally.
        env.set_unknown_method_callback(unknown_method_callback);
        // `raise_exception` is referenced by the Llama 2 template
        // when the user/assistant alternation is broken. Bind it
        // so rendering surfaces a clear template error rather
        // than `UnknownFunction`.
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        // `strftime_now` appears in some Llama 3.2 templates
        // (date stamps in system messages). Stub it with a
        // deterministic placeholder so the template renders;
        // the date is irrelevant to the M5 demo path and a real
        // implementation can be wired post-MVP.
        env.add_function("strftime_now", |_fmt: String| -> Result<String, minijinja::Error> {
            Ok(String::from("01 January 2026"))
        });

        env.add_template("chat", template_src)
            .map_err(|e| TokenizerError::Template(format!("compile: {e}")))?;
        let tmpl = env.get_template("chat")
            .map_err(|e| TokenizerError::Template(format!("get: {e}")))?;

        let rendered = tmpl.render(context! {
            messages => messages,
            // Templates that don't reference `bos_token` won't
            // care; ones that do (Llama 2) get the literal
            // string. Empty when the model has no global BOS.
            bos_token => self.bos_token.clone().unwrap_or_default(),
            eos_token => self.eos_token.clone(),
            add_generation_prompt => true,
        })
        .map_err(|e| TokenizerError::Template(format!("render: {e}")))?;

        Ok(rendered)
    }

    /// BOS token id, or `None` when the model has no global
    /// BOS sentinel (Qwen 2.5 case). 1 for Llama 2.
    pub fn bos_id(&self) -> Option<u32> { self.bos_id }
    /// EOS token id (2 for Llama 2, model-dependent otherwise).
    pub fn eos_id(&self) -> u32 { self.eos_id }
    /// Whether the underlying config wants BOS prepended by default.
    pub fn add_bos_token(&self) -> bool { self.add_bos_token }
    /// Whether the underlying config wants EOS appended by default.
    pub fn add_eos_token(&self) -> bool { self.add_eos_token }
    /// True if the model ships a chat template.
    pub fn has_chat_template(&self) -> bool { self.chat_template.is_some() }
    /// Underlying vocabulary size (= tokenizer.json vocab len).
    pub fn vocab_size(&self) -> usize { self.inner.get_vocab_size(true) }
    /// Path the tokenizer was loaded from.
    pub fn model_dir(&self) -> &Path { &self.model_dir }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path resolution helper: tests run from the workspace
    /// root, so model dirs live at `models/<name>/`. If the
    /// path doesn't exist (CI without checkpoints) the test
    /// short-circuits to `Ok(())` after printing a skip note —
    /// matches the existing M4.6 / M4.7 family-test pattern.
    fn model_dir(name: &str) -> Option<PathBuf> {
        let p = PathBuf::from("models").join(name);
        if p.exists() { Some(p) } else { None }
    }

    #[test]
    fn llama2_loads_and_reports_known_ids() {
        let Some(dir) = model_dir("llama-2-13b-chat") else {
            eprintln!("[skip] llama-2-13b-chat not present");
            return;
        };
        let tok = AteniaTokenizer::from_model_dir(&dir).unwrap();
        assert_eq!(tok.bos_id(), Some(1), "Llama 2 BOS = 1 (<s>)");
        assert_eq!(tok.eos_id(), 2, "Llama 2 EOS = 2 (</s>)");
        assert_eq!(tok.vocab_size(), 32_000);
        assert!(tok.has_chat_template());
    }

    #[test]
    fn llama2_round_trip_preserves_text() {
        let Some(dir) = model_dir("llama-2-13b-chat") else { return; };
        let tok = AteniaTokenizer::from_model_dir(&dir).unwrap();
        // D66 round-trip: decode(encode(x)) == x for clean text.
        // BOS is stripped by skip_special_tokens. Whitespace
        // normalisation in SentencePiece can collapse leading
        // spaces; we test inputs that don't trigger that.
        let cases = ["Hello, my name is",
                     "The capital of France is Paris.",
                     "1234567890",
                     "End of file."];
        for text in &cases {
            let ids = tok.encode(text, true).unwrap();
            let back = tok.decode(&ids, true).unwrap();
            // Llama 2 tokenizer adds a leading space when
            // decoding; trim_start to compare semantic content.
            assert_eq!(back.trim_start(), *text, "round-trip mismatch on {text:?}");
        }
    }

    #[test]
    fn llama2_encode_known_prompt_ids() {
        // R1 falsifier — locked IDs cross-checked against the
        // Python reference:
        //   from transformers import AutoTokenizer
        //   t = AutoTokenizer.from_pretrained(".../llama-2-13b-chat")
        //   t.encode("Hello, my name is")
        //   -> [1, 15043, 29892, 590, 1024, 338]
        let Some(dir) = model_dir("llama-2-13b-chat") else { return; };
        let tok = AteniaTokenizer::from_model_dir(&dir).unwrap();
        let ids = tok.encode("Hello, my name is", true).unwrap();
        assert_eq!(ids, vec![1, 15043, 29892, 590, 1024, 338],
            "byte-exact ID match vs HF transformers reference");
    }

    #[test]
    fn llama2_chat_template_renders() {
        let Some(dir) = model_dir("llama-2-13b-chat") else { return; };
        let tok = AteniaTokenizer::from_model_dir(&dir).unwrap();
        let prompt = tok.apply_chat_template(&[
            ChatMessage::user("Hi!"),
        ]).unwrap();
        // Llama 2 chat template wraps user messages in
        // `<s>[INST] ... [/INST]`. We assert structural shape,
        // not exact whitespace, so template tweaks upstream
        // don't break us.
        assert!(prompt.contains("[INST]"), "missing [INST] in {prompt:?}");
        assert!(prompt.contains("Hi!"), "missing user content in {prompt:?}");
        assert!(prompt.contains("[/INST]"), "missing [/INST] in {prompt:?}");
    }

    #[test]
    fn family_models_load_when_present() {
        // Skip-friendly smoke test across the whole M5 model
        // scope: every family must load + encode + decode + (if
        // it has a chat template) render. Doesn't lock IDs —
        // each family ships its own vocabulary; R1 byte-exact
        // ID match per family lives in dedicated tests.
        for name in &[
            "tinyllama-1.1b",
            "llama-3.2-1b-instruct",
            "qwen2.5-1.5b-instruct",
            "smollm2-1.7b-instruct",
        ] {
            let Some(dir) = model_dir(name) else {
                eprintln!("[skip] {name} not present");
                continue;
            };
            let tok = AteniaTokenizer::from_model_dir(&dir)
                .unwrap_or_else(|e| panic!("{name}: load: {e}"));
            let ids = tok.encode("Hello world", false)
                .unwrap_or_else(|e| panic!("{name}: encode: {e}"));
            assert!(!ids.is_empty(), "{name}: encode produced no tokens");
            let _back = tok.decode(&ids, true)
                .unwrap_or_else(|e| panic!("{name}: decode: {e}"));
            if tok.has_chat_template() {
                let _prompt = tok.apply_chat_template(&[ChatMessage::user("Hi!")])
                    .unwrap_or_else(|e| panic!("{name}: chat template: {e}"));
            }
        }
    }
}
