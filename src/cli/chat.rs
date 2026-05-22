//! **CLI interactive chat (CLI-4): `atenia chat`.**
//!
//! A simple interactive REPL on top of the existing generation
//! pipeline. It keeps an in-memory conversation history, applies
//! the model's chat template when one is available, and reuses
//! `GenerationPipeline::generate_raw` unchanged — no engine code,
//! no graph builder, no generation logic is touched.
//!
//! Output contract: **stdout** carries model responses (and the
//! `/history` dump); **stderr** carries the `You> ` turn prompt,
//! banners, the "Thinking ..." indicator, logs and errors. Piping
//! `atenia chat` therefore yields a clean transcript of assistant
//! turns. A non-interactive (piped) invocation skips the welcome
//! banner via TTY detection.
//!
//! The pipeline is **lazy-loaded** on the first real message, so
//! `/exit` / `/help` / `/reset` / `/history` work — and are
//! testable — without a real checkpoint on disk.

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use crate::nn::llama::{GenerationPipeline, StdoutTokenSink};
use crate::tokenizer::ChatMessage as TokChatMessage;

use super::error::CliError;
use super::logging;

/// Who authored a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatRole {
    /// A message typed by the user.
    User,
    /// A message produced by the model.
    Assistant,
}

/// One conversation turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    /// Who authored this turn.
    pub role: ChatRole,
    /// The turn text.
    pub content: String,
}

/// The in-memory conversation state.
#[derive(Debug, Default)]
pub struct ChatSession {
    /// Every turn so far, oldest first.
    pub history: Vec<ChatMessage>,
}

impl ChatSession {
    /// A fresh, empty session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a user turn.
    pub fn push_user(&mut self, content: String) {
        self.history.push(ChatMessage {
            role: ChatRole::User,
            content,
        });
    }

    /// Append an assistant turn.
    pub fn push_assistant(&mut self, content: String) {
        self.history.push(ChatMessage {
            role: ChatRole::Assistant,
            content,
        });
    }

    /// Drop every turn.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Render the history for the `/history` command.
    pub fn render_history(&self) -> String {
        if self.history.is_empty() {
            return "(history is empty)".to_string();
        }
        let mut out = String::new();
        for m in &self.history {
            let who = match m.role {
                ChatRole::User => "User",
                ChatRole::Assistant => "Assistant",
            };
            out.push_str(&format!("{who}: {}\n", m.content));
        }
        out
    }

    /// Convert the history to the tokenizer's `ChatMessage` form,
    /// for `apply_chat_template`.
    fn to_tokenizer_messages(&self) -> Vec<TokChatMessage> {
        self.history
            .iter()
            .map(|m| match m.role {
                ChatRole::User => TokChatMessage::user(m.content.clone()),
                ChatRole::Assistant => TokChatMessage::assistant(m.content.clone()),
            })
            .collect()
    }

    /// Build the fallback prompt used when the model ships no chat
    /// template: a plain `User:` / `Assistant:` transcript ending
    /// with an open `Assistant:` turn for the model to complete.
    pub fn render_fallback_prompt(&self) -> String {
        let mut out = String::new();
        for m in &self.history {
            match m.role {
                ChatRole::User => out.push_str(&format!("User: {}\n", m.content)),
                ChatRole::Assistant => {
                    out.push_str(&format!("Assistant: {}\n", m.content))
                }
            }
        }
        out.push_str("Assistant:");
        out
    }
}

/// A recognised slash command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatCommand {
    /// `/exit` — leave the session.
    Exit,
    /// `/reset` — clear the history.
    Reset,
    /// `/history` — print the history.
    History,
    /// `/help` — list the commands.
    Help,
    /// An unrecognised `/...` token.
    Unknown,
}

/// Parse a slash command. Returns `None` for ordinary input.
pub fn parse_command(line: &str) -> Option<ChatCommand> {
    let t = line.trim();
    if !t.starts_with('/') {
        return None;
    }
    Some(match t {
        "/exit" | "/quit" => ChatCommand::Exit,
        // `/clear` is an alias for `/reset` — both clear history.
        "/reset" | "/clear" => ChatCommand::Reset,
        "/history" => ChatCommand::History,
        "/help" => ChatCommand::Help,
        _ => ChatCommand::Unknown,
    })
}

/// Arguments for `atenia chat`.
pub struct ChatArgs {
    /// Model directory (a `config.json` + weights, or a `.gguf`).
    pub model: PathBuf,
    /// Maximum new tokens per assistant turn.
    pub max_tokens: usize,
    /// Sampling temperature. Accepted for forward compatibility;
    /// generation is currently greedy, so a non-zero value only
    /// emits a warning (wiring sampling needs engine support and
    /// is out of CLI-4 scope).
    pub temperature: f32,
    /// Skip the model chat template; use the `User:`/`Assistant:`
    /// fallback transcript instead.
    pub no_chat_template: bool,
}

const HELP_TEXT: &str = "\
Commands:
  /help     show this help
  /history  print the conversation so far
  /reset    clear the conversation history
  /clear    alias for /reset
  /exit     leave the chat (Ctrl+D also works)
Anything else is sent to the model as your message.";

/// Run the interactive chat REPL. Returns the process exit code.
pub fn run_chat(args: ChatArgs) -> i32 {
    logging::info("command start: chat");

    // ---- up-front argument validation ----
    if !args.model.exists() {
        let err = CliError::io_not_found("the model directory", &args.model);
        eprintln!("{err}");
        return err.exit.code();
    }
    if args.model.is_file() {
        // A `--model` pointing at a file (e.g. an adapter spec) is
        // a user error: chat needs a model directory with weights.
        let err = CliError::invalid_args(
            "--model must be a model directory, not a file",
            "Point --model at a directory containing config.json + weights, \
             or a directory with a single .gguf file. To validate an adapter \
             spec file, use `atenia load <file>` instead.",
        );
        eprintln!("{err}");
        return err.exit.code();
    }
    if args.max_tokens == 0 {
        let err = CliError::invalid_args(
            "--max-tokens must be greater than 0",
            "Pass --max-tokens with a value of 1 or more.",
        );
        eprintln!("{err}");
        return err.exit.code();
    }
    if args.temperature > 0.0 {
        logging::warn(
            "temperature is not yet supported; chat uses greedy decoding. \
             The --temperature value is ignored.",
        );
    }

    let quiet = !logging::level_at_least(logging::LogLevel::Warn);
    // TTY detection: a piped invocation (`echo ... | atenia chat`)
    // is non-interactive — skip the welcome banner, which is chrome
    // a script does not need. The `You> ` turn prompt is still
    // printed (to stderr) so a piped transcript stays readable.
    let interactive = std::io::stdin().is_terminal();
    if !quiet && interactive {
        eprintln!("Atenia chat — type /help for commands, /exit to leave.");
    }

    let mut session = ChatSession::new();
    // Lazy: the (slow) model load happens on the first message,
    // so the slash commands work without a real checkpoint.
    let mut pipeline: Option<GenerationPipeline> = None;

    let stdin = std::io::stdin();
    let mut handle = stdin.lock();
    let mut line = String::new();

    loop {
        // The turn prompt is UI chrome → stderr, so a piped stdout
        // stays a clean transcript of assistant turns.
        eprint!("You> ");
        let _ = std::io::stderr().flush();

        line.clear();
        match handle.read_line(&mut line) {
            Ok(0) => {
                // EOF (Ctrl+D): leave gracefully.
                if !quiet {
                    eprintln!();
                    eprintln!("(end of input)");
                }
                break;
            }
            Ok(_) => {}
            Err(e) => {
                let err = CliError::from(e);
                eprintln!("{err}");
                return err.exit.code();
            }
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // ---- slash commands ----
        if let Some(cmd) = parse_command(input) {
            match cmd {
                ChatCommand::Exit => break,
                ChatCommand::Reset => {
                    session.reset();
                    if !quiet {
                        eprintln!("(history cleared)");
                    }
                }
                ChatCommand::History => {
                    // The history dump is requested output → stdout.
                    print!("{}", session.render_history());
                    let _ = std::io::stdout().flush();
                }
                ChatCommand::Help => {
                    if !quiet {
                        eprintln!("{HELP_TEXT}");
                    }
                }
                ChatCommand::Unknown => {
                    eprintln!("unknown command `{input}` — type /help");
                }
            }
            continue;
        }

        // ---- a real user message ----
        session.push_user(input.to_string());

        // Lazy-load the pipeline on first use.
        if pipeline.is_none() {
            logging::info("loading model ...");
            match load_pipeline(&args.model) {
                Ok(p) => pipeline = Some(p),
                Err(err) => {
                    eprintln!("{err}");
                    return err.exit.code();
                }
            }
        }
        let pipe = pipeline.as_ref().expect("pipeline loaded above");

        // Build the prompt: the model chat template when available
        // (and not disabled), else the plain transcript fallback.
        let prompt = match build_prompt(pipe, &session, args.no_chat_template) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("{err}");
                return err.exit.code();
            }
        };
        logging::debug(&format!("full prompt:\n{prompt}"));
        logging::info("generating response ...");
        // A visible "thinking" indicator (stderr) so the user is
        // not left staring at a silent gap during prefill. Shown
        // by default; suppressed only under --quiet.
        if !quiet {
            eprintln!("Thinking ...");
        }

        // Stream the response to stdout; `generate_raw` also
        // returns the full text, which we keep for the history.
        let mut sink = StdoutTokenSink;
        match pipe.generate_raw(&prompt, args.max_tokens, &mut sink) {
            Ok(text) => {
                println!();
                session.push_assistant(text);
            }
            Err(e) => {
                let err =
                    CliError::generation_failed("generation failed", e.to_string());
                eprintln!("{err}");
                return err.exit.code();
            }
        }
    }

    logging::info("command completed: chat");
    0
}

/// Load the generation pipeline, detecting a GGUF directory the
/// same way `atenia generate` does. Translates a load failure into
/// a [`CliError`].
fn load_pipeline(model_dir: &Path) -> Result<GenerationPipeline, CliError> {
    let is_gguf = model_dir
        .read_dir()
        .ok()
        .map(|entries| {
            entries.flatten().any(|e| {
                e.path().extension().and_then(|x| x.to_str()) == Some("gguf")
            })
        })
        .unwrap_or(false);

    let result = if is_gguf {
        GenerationPipeline::from_model_dir_with_options(model_dir, true)
    } else {
        GenerationPipeline::from_model_dir(model_dir)
    };
    result.map_err(|e| CliError::generation_failed("failed to load the model", e.to_string()))
}

/// Build the prompt string for the current turn.
fn build_prompt(
    pipe: &GenerationPipeline,
    session: &ChatSession,
    no_chat_template: bool,
) -> Result<String, CliError> {
    if !no_chat_template && pipe.tokenizer.has_chat_template() {
        pipe.tokenizer
            .apply_chat_template(&session.to_tokenizer_messages())
            .map_err(|e| {
                CliError::generation_failed("failed to apply the chat template", e.to_string())
            })
    } else {
        Ok(session.render_fallback_prompt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_command_recognises_slash_commands() {
        assert_eq!(parse_command("/exit"), Some(ChatCommand::Exit));
        assert_eq!(parse_command("/quit"), Some(ChatCommand::Exit));
        assert_eq!(parse_command("/reset"), Some(ChatCommand::Reset));
        assert_eq!(parse_command("/history"), Some(ChatCommand::History));
        assert_eq!(parse_command("/help"), Some(ChatCommand::Help));
        assert_eq!(parse_command("/bogus"), Some(ChatCommand::Unknown));
    }

    #[test]
    fn parse_command_ignores_ordinary_input() {
        assert_eq!(parse_command("hello there"), None);
        assert_eq!(parse_command("what is /etc?"), None);
    }

    #[test]
    fn session_push_and_reset() {
        let mut s = ChatSession::new();
        s.push_user("hi".into());
        s.push_assistant("hello".into());
        assert_eq!(s.history.len(), 2);
        s.reset();
        assert!(s.history.is_empty());
    }

    #[test]
    fn render_history_empty_and_populated() {
        let mut s = ChatSession::new();
        assert!(s.render_history().contains("empty"));
        s.push_user("hi".into());
        s.push_assistant("hello".into());
        let h = s.render_history();
        assert!(h.contains("User: hi"));
        assert!(h.contains("Assistant: hello"));
    }

    #[test]
    fn fallback_prompt_ends_with_open_assistant_turn() {
        let mut s = ChatSession::new();
        s.push_user("hi".into());
        let p = s.render_fallback_prompt();
        assert!(p.starts_with("User: hi\n"));
        assert!(p.ends_with("Assistant:"));
    }

    #[test]
    fn tokenizer_messages_preserve_roles() {
        let mut s = ChatSession::new();
        s.push_user("q".into());
        s.push_assistant("a".into());
        let msgs = s.to_tokenizer_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
    }
}
