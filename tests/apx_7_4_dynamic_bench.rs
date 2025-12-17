use atenia_engine::apx7::dynamic_load::{sample_system_load, choose_strategy};

#[test]
fn apx_7_4_dynamic_bench() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.4");
    }

    let snap = sample_system_load();
    let strategy = choose_strategy(&snap);

    println!(
        "[APX 7.4] load={:.1}% threads={} -> strategy={}",
        snap.cpu_load,
        snap.threads_available,
        strategy,
    );
}
