use std::process::Command;

#[test]
fn cli_explain_smoke_test() {
    // Use the compiled binary through the CARGO_BIN_EXE_atenia env var.
    let bin_path = env!("CARGO_BIN_EXE_atenia");

    let output = Command::new(bin_path)
        .args([
            "explain",
            "--gpu-available=true",
            "--vram-band=0",
            "--ram-band=0",
        ])
        .output()
        .expect("failed to run atenia explain");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Atenia selected")
            || stdout.contains("No learned data available for the given context."),
        "unexpected CLI output: {}",
        stdout
    );
}
