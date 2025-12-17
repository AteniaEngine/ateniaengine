#[allow(unexpected_cfgs)]
#[test]
fn nightly_active_for_bench() {
    // En modo normal (sin bench_nightly) no fallamos, solo informamos opcionalmente.
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
