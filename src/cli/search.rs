//! **AQS-10** — `atenia search` experimental CLI command.
//!
//! Renders an AQS certification report + manifest draft from a
//! **pre-computed end-to-end results file**. The CLI deliberately does
//! NOT load a model, run a forward, use test fixtures, or simulate an F64
//! reference — AQS end-to-end certification requires a per-model F64
//! reference that only the heavy harness can produce. This command is the
//! honest, decoupled front-end: feed it the harness output and it
//! classifies / ranks / certifies / renders.
//!
//! ## Usage
//!
//! ```text
//! atenia search --results aqs-results.json --report
//! atenia search --results aqs-results.json --report --manifest
//! ```
//!
//! ## Results file schema
//!
//! ```json
//! {
//!   "model": "tinyllama-1.1b",
//!   "baseline_memory_bytes": 2260729856,
//!   "results": [
//!     {
//!       "candidate_name": "bf16",
//!       "max_abs_diff": 0.000063,
//!       "mean_abs_diff": 0.000008,
//!       "rmse": 0.00001,
//!       "argmax_match": true,
//!       "memory_bytes": 2260729856
//!     }
//!   ]
//! }
//! ```
//!
//! ## Scope (AQS-10)
//!
//! No model loading, no forward, no test fixtures, no F64 simulation, no
//! runtime / numcert manifest integration, no CUDA, no new techniques.
//! Experimental, CPU-only, opt-in. The manifest output is the AQS-6
//! `3.0.0-draft` string, never a productive manifest.

use std::path::PathBuf;

use serde::Deserialize;

use crate::cli::error::CliError;
use crate::quant::certification::AqsCertificationReport;
use crate::quant::end_to_end::EndToEndEvalResult;
use crate::quant::search::search_from_end_to_end_results;

/// Parsed `atenia search` arguments (mirrors the clap layer in the
/// binary so the binary stays thin).
#[derive(Debug, Clone)]
pub struct SearchArgs {
    /// Path to the pre-computed end-to-end results JSON. Required —
    /// without it the command cannot do anything honest.
    pub results: Option<PathBuf>,
    /// Print the human-readable report table.
    pub report: bool,
    /// Print the experimental `3.0.0-draft` manifest.
    pub manifest: bool,
    /// Reserved flag — when a future model-driving CLI exists, this would
    /// opt into evaluating real GPTQ (~hours). AQS-10 only parses it (the
    /// results-file path never runs GPTQ); it is surfaced so the flag is
    /// stable and documented.
    pub include_gptq: bool,
}

/// On-disk results-file schema.
#[derive(Debug, Deserialize)]
struct ResultsFile {
    model: String,
    #[serde(default)]
    baseline_memory_bytes: Option<u64>,
    results: Vec<ResultEntry>,
}

/// One result entry — the serialisable mirror of [`EndToEndEvalResult`].
#[derive(Debug, Deserialize)]
struct ResultEntry {
    candidate_name: String,
    max_abs_diff: f32,
    mean_abs_diff: f32,
    rmse: f32,
    argmax_match: bool,
    memory_bytes: u64,
}

impl ResultEntry {
    fn into_eval(self) -> EndToEndEvalResult {
        EndToEndEvalResult {
            candidate_name: self.candidate_name,
            max_abs_diff: self.max_abs_diff,
            mean_abs_diff: self.mean_abs_diff,
            rmse: self.rmse,
            argmax_match: self.argmax_match,
            memory_bytes: self.memory_bytes,
        }
    }
}

/// Build an [`AqsCertificationReport`] from a results file's bytes.
/// Factored out of the I/O so it is unit-testable without a filesystem.
fn report_from_json(json: &str) -> Result<AqsCertificationReport, CliError> {
    let parsed: ResultsFile = serde_json::from_str(json).map_err(|e| {
        CliError::invalid_args(
            "the AQS results file is not valid JSON",
            "Make sure --results points at a file produced by the AQS \
             end-to-end harness (see `docs/HANDOFF_AQS_10.md` for the schema).",
        )
        .with_detail("parse error", e.to_string())
    })?;

    let results: Vec<EndToEndEvalResult> =
        parsed.results.into_iter().map(ResultEntry::into_eval).collect();

    let search = search_from_end_to_end_results(
        &parsed.model,
        crate::quant::certification::ADR_004_GATE,
        parsed.baseline_memory_bytes,
        results,
    );
    Ok(search.report)
}

/// `atenia search` entry point. Returns a process exit code.
pub fn run_search(args: SearchArgs) -> i32 {
    let Some(results_path) = args.results.as_ref() else {
        let err = CliError::invalid_args(
            "atenia search requires --results",
            "AQS end-to-end certification needs a per-model F64 reference, \
             which only the end-to-end harness can produce. Run the harness \
             to generate a results file, then:\n  \
             atenia search --results aqs-results.json --report --manifest",
        );
        eprintln!("{}", err.render());
        return err.exit.code();
    };

    if !results_path.is_file() {
        let err = CliError::io_not_found("the AQS results file", results_path);
        eprintln!("{}", err.render());
        return err.exit.code();
    }

    let json = match std::fs::read_to_string(results_path) {
        Ok(s) => s,
        Err(e) => {
            let err = CliError::invalid_args(
                "the AQS results file could not be read",
                "Check the path and file permissions.",
            )
            .with_detail("path", results_path.display().to_string())
            .with_detail("io error", e.to_string());
            eprintln!("{}", err.render());
            return err.exit.code();
        }
    };

    let report = match report_from_json(&json) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("{}", err.render());
            return err.exit.code();
        }
    };

    // Default to showing the report when neither flag is given — an empty
    // run would be useless.
    let show_report = args.report || !args.manifest;

    println!("AQS Search Report\n");
    println!("model: {}", report.model_name);
    println!(
        "best certified   : {}",
        report.best_certified().unwrap_or("(none)")
    );
    println!(
        "best useful lossy: {}",
        report.best_useful_lossy().unwrap_or("(none)")
    );
    println!();

    if show_report {
        print!("{}", report.render_human_report());
    }

    if args.manifest {
        println!("\nManifest Draft\n");
        print!("{}", report.render_manifest_draft());
    }

    // Note: include_gptq is parsed/accepted but does nothing on the
    // results-file path (no model is run here). Surface it honestly.
    if args.include_gptq {
        println!(
            "\nnote: --include-gptq has no effect on the results-file path \
             (atenia search does not run a model; GPTQ rows, if any, come \
             from the results file itself)."
        );
    }

    crate::cli::exit::CliExit::Success.code()
}

// ============================================================================
// Tests (no model, no fixture, no I/O for the core path)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "model": "tinyllama-1.1b",
        "baseline_memory_bytes": 2260729856,
        "results": [
            {"candidate_name":"bf16","max_abs_diff":0.000063,"mean_abs_diff":0.000008,"rmse":0.00001,"argmax_match":true,"memory_bytes":2260729856},
            {"candidate_name":"awq_g128_a0.25","max_abs_diff":0.889217,"mean_abs_diff":0.074738,"rmse":0.105758,"argmax_match":true,"memory_bytes":1165688832},
            {"candidate_name":"plain_int8","max_abs_diff":1.260771,"mean_abs_diff":0.145249,"rmse":0.202987,"argmax_match":false,"memory_bytes":1165688832}
        ]
    }"#;

    #[test]
    fn report_from_json_classifies_and_selects() {
        let report = report_from_json(SAMPLE).unwrap();
        assert_eq!(report.model_name, "tinyllama-1.1b");
        assert_eq!(report.best_certified(), Some("bf16"));
        assert_eq!(report.best_useful_lossy(), Some("awq_g128_a0.25"));
    }

    #[test]
    fn report_human_render_is_deterministic() {
        let report = report_from_json(SAMPLE).unwrap();
        let a = report.render_human_report();
        let b = report.render_human_report();
        assert_eq!(a, b);
        assert!(a.contains("bf16"));
        assert!(a.contains("awq_g128_a0.25"));
    }

    #[test]
    fn report_manifest_render_is_draft() {
        let report = report_from_json(SAMPLE).unwrap();
        let m = report.render_manifest_draft();
        assert!(m.contains("schema_version: \"3.0.0-draft\""));
        assert!(m.contains("model: tinyllama-1.1b"));
        assert!(m.contains("certified_policy: bf16"));
        // AWQ must never be certified.
        assert!(m.contains("status: useful_lossy"));
    }

    #[test]
    fn invalid_json_is_a_clean_error() {
        let err = report_from_json("{ not valid json").unwrap_err();
        let rendered = err.render();
        assert!(rendered.contains("E-CLI-INVALID-ARGS"));
        assert!(rendered.contains("not valid JSON"));
    }

    #[test]
    fn missing_baseline_yields_no_compression_ratio() {
        let json = r#"{
            "model":"m",
            "results":[
                {"candidate_name":"bf16","max_abs_diff":0.0,"mean_abs_diff":0.0,"rmse":0.0,"argmax_match":true,"memory_bytes":100}
            ]
        }"#;
        let report = report_from_json(json).unwrap();
        assert_eq!(report.baseline_memory_bytes, None);
        assert!(report.policy("bf16").unwrap().compression_ratio.is_none());
    }

    #[test]
    fn run_search_without_results_is_user_error() {
        let code = run_search(SearchArgs {
            results: None,
            report: true,
            manifest: false,
            include_gptq: false,
        });
        // CliExit::UserInput == 2.
        assert_eq!(code, 2);
    }

    #[test]
    fn run_search_missing_file_is_user_error() {
        let code = run_search(SearchArgs {
            results: Some(PathBuf::from("definitely/does/not/exist-aqs.json")),
            report: true,
            manifest: false,
            include_gptq: false,
        });
        assert_eq!(code, 2);
    }

    #[test]
    fn include_gptq_flag_parses_and_is_inert_on_results_path() {
        // Round-trips through a temp file; should succeed (exit 0) and not
        // change classification.
        let dir = std::env::temp_dir();
        let path = dir.join("atenia_aqs10_test_results.json");
        std::fs::write(&path, SAMPLE).unwrap();
        let code = run_search(SearchArgs {
            results: Some(path.clone()),
            report: true,
            manifest: true,
            include_gptq: true,
        });
        let _ = std::fs::remove_file(&path);
        assert_eq!(code, 0);
    }
}
