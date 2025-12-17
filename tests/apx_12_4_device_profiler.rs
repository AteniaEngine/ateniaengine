use atenia_engine::gpu::device::device_profile;

#[test]
fn apx_12_4_device_profiler_basic() {
    let p = device_profile();
    if std::env::var("ATENIA_DEBUG").ok().as_deref() == Some("1") {
        println!("[PROFILE] SMs={}", p.sm_count);
        println!("[PROFILE] max_threads_per_sm={}", p.max_threads_per_sm);
        println!("[PROFILE] max_threads_per_block={}", p.max_threads_per_block);
        println!("[PROFILE] warp_size={}", p.warp_size);
        println!("[PROFILE] shared_mem_per_sm={}", p.shared_mem_per_sm);
        println!("[PROFILE] regs_per_sm={}", p.max_registers_per_sm);
    }

    // CPU fallback: the profiler should still return sane defaults
    assert!(p.warp_size >= 1);
    assert!(p.sm_count >= 0);                // may be 0 if GPU unreachable
    assert!(p.max_threads_per_block >= 1);   // defaults must be >= 1
}
