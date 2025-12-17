use atenia_engine::apx7::dynamic_load::{LoadSnapshot, LAST_SNAPSHOT, choose_strategy};

#[test]
fn apx_7_4_prefers_seq_under_heavy_load() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.4");
    }

    {
        let mut guard = LAST_SNAPSHOT.write().unwrap();
        *guard = LoadSnapshot { cpu_load: 95.0, threads_available: 1 };
    }

    let snap = atenia_engine::apx7::dynamic_load::get_last_snapshot();
    let strategy = choose_strategy(&snap);
    assert_eq!(strategy, "seq");
}

#[test]
fn apx_7_4_prefers_ws_when_many_threads_available() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.4");
    }

    {
        let mut guard = LAST_SNAPSHOT.write().unwrap();
        *guard = LoadSnapshot { cpu_load: 20.0, threads_available: 20 };
    }

    let snap = atenia_engine::apx7::dynamic_load::get_last_snapshot();
    let strategy = choose_strategy(&snap);
    assert_eq!(strategy, "ws");
}

#[test]
fn apx_7_4_prefers_pex_when_limited_threads() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.4");
    }

    {
        let mut guard = LAST_SNAPSHOT.write().unwrap();
        *guard = LoadSnapshot { cpu_load: 40.0, threads_available: 2 };
    }

    let snap = atenia_engine::apx7::dynamic_load::get_last_snapshot();
    let strategy = choose_strategy(&snap);
    assert_eq!(strategy, "pex");
}
