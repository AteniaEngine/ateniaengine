use std::env;
use std::process;

use atenia_engine::v13::learning_narrative::build_narrative;
use atenia_engine::v13::learning_snapshot::LearningContextSnapshot;
use atenia_engine::v13::self_trainer::SelfTrainer;

fn print_usage() {
    eprintln!(
        "Usage: atenia explain --gpu-available=<true|false> --vram-band=<0..3> --ram-band=<0..3>",
    );
}

fn parse_bool_flag(value: &str) -> Option<bool> {
    match value {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_u8_band(value: &str) -> Option<u8> {
    match value.parse::<u8>() {
        Ok(v) if v <= 3 => Some(v),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 || args.get(1).map(String::as_str) != Some("explain") {
        print_usage();
        process::exit(1);
    }

    let mut gpu_available_opt: Option<bool> = None;
    let mut vram_band_opt: Option<u8> = None;
    let mut ram_band_opt: Option<u8> = None;

    for arg in &args[2..] {
        let mut parts = arg.splitn(2, '=');
        let key = parts.next().unwrap_or("");
        let value = parts.next();

        if value.is_none() {
            print_usage();
            process::exit(1);
        }
        let value = value.unwrap();

        match key {
            "--gpu-available" => {
                match parse_bool_flag(value) {
                    Some(v) => gpu_available_opt = Some(v),
                    None => {
                        print_usage();
                        process::exit(1);
                    }
                }
            }
            "--vram-band" => {
                match parse_u8_band(value) {
                    Some(v) => vram_band_opt = Some(v),
                    None => {
                        print_usage();
                        process::exit(1);
                    }
                }
            }
            "--ram-band" => {
                match parse_u8_band(value) {
                    Some(v) => ram_band_opt = Some(v),
                    None => {
                        print_usage();
                        process::exit(1);
                    }
                }
            }
            _ => {
                print_usage();
                process::exit(1);
            }
        }
    }

    let gpu_available = match gpu_available_opt {
        Some(v) => v,
        None => {
            print_usage();
            process::exit(1);
        }
    };
    let vram_band = match vram_band_opt {
        Some(v) => v,
        None => {
            print_usage();
            process::exit(1);
        }
    };
    let ram_band = match ram_band_opt {
        Some(v) => v,
        None => {
            print_usage();
            process::exit(1);
        }
    };

    let ctx = LearningContextSnapshot {
        gpu_available,
        vram_band,
        ram_band,
    };

    let trainer = SelfTrainer::new();

    match trainer.explain_decision(ctx) {
        Some(text) => {
            // Structured explanation should be available for the same context if
            // textual explanation exists; unwrap here is acceptable at the CLI
            // layer.
            let structured = match trainer.explain_decision_structured(ctx) {
                Some(s) => s,
                None => {
                    println!("No learned data available for the given context.");
                    process::exit(0);
                }
            };

            let narrative = build_narrative(&text, &structured);
            println!("{}", narrative.narrative);
        }
        None => {
            println!("No learned data available for the given context.");
        }
    }
}
