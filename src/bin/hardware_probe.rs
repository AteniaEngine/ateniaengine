//! Hardware probe binary.
//!
//! Usage:
//!     hardware_probe [--output text|json] [--help]
//!
//! Default output is `text` (human-readable). `--output json` emits a
//! machine-readable JSON document suitable for piping into jq or into
//! a config generator.
//!
//! Build with:
//!     cargo build --release --bin hardware_probe --features hw-probe
//!
//! See `docs/HARDWARE_PROBE.md` for field documentation and
//! multi-vendor support notes.

use atenia_engine::hw_probe;

fn print_usage() {
    eprintln!(
        "Usage: hardware_probe [--output text|json]\n\
         \n\
         Options:\n\
           --output <text|json>   Output format (default: text)\n\
           --help                 Print this message and exit"
    );
}

enum OutputFormat {
    Text,
    Json,
}

fn parse_args() -> Result<OutputFormat, String> {
    let mut args = std::env::args().skip(1);
    let mut format = OutputFormat::Text;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--output" => {
                let value = args.next().ok_or_else(|| {
                    "--output requires a value (text or json)".to_string()
                })?;
                format = match value.as_str() {
                    "text" => OutputFormat::Text,
                    "json" => OutputFormat::Json,
                    other => {
                        return Err(format!(
                            "invalid --output value '{}'; expected text or json",
                            other
                        ));
                    }
                };
            }
            other => {
                return Err(format!("unknown argument '{}'", other));
            }
        }
    }

    Ok(format)
}

fn main() {
    let format = match parse_args() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {}", e);
            eprintln!();
            print_usage();
            std::process::exit(2);
        }
    };

    let report = hw_probe::probe();

    match format {
        OutputFormat::Text => {
            print!("{}", report);
        }
        OutputFormat::Json => {
            // Pretty-print so humans can still read the JSON output,
            // at the cost of a few extra bytes. Tools that need
            // compact form can pipe through `jq -c`.
            match serde_json::to_string_pretty(&report) {
                Ok(s) => println!("{}", s),
                Err(e) => {
                    eprintln!("error: failed to serialize report as JSON: {}", e);
                    std::process::exit(3);
                }
            }
        }
    }
}
