#[allow(unexpected_cfgs)]
#[test]
fn nightly_active_for_bench() {
    // In normal mode (without bench_nightly) we do not fail, we only optionally report.
    #[cfg(not(bench_nightly))]
    {
        if atenia_engine::apx_debug_enabled() {
            eprintln!("Nightly bench features not active (bench_nightly cfg unset). Test is a no-op.");
        }
    }

    // Cuando se compila con `--cfg bench_nightly` (p.ej. en benches), confirmamos estado.
    #[cfg(bench_nightly)]
    {
        eprintln!("Nightly OK — benchmark features active.");
    }
}
