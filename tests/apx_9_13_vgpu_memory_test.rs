use atenia_engine::VGpuMemory;

#[test]
fn apx_9_13_global_memory_rw() {
    let mut m = VGpuMemory::new(64, 16, 2, 8);
    m.store_global(10, 3.5);
    assert_eq!(m.load_global(10), 3.5);
}

#[test]
fn apx_9_13_shared_memory_rw() {
    let mut m = VGpuMemory::new(64, 16, 2, 8);
    m.store_shared(1, 4, 7.25);
    assert_eq!(m.load_shared(1, 4), 7.25);
}

#[test]
fn apx_9_13_local_memory_rw() {
    let mut m = VGpuMemory::new(64, 16, 2, 8);
    m.store_local(5, 3, 9.0);
    assert_eq!(m.load_local(5, 3), 9.0);
}

#[test]
fn apx_9_13_memory_layout_structure() {
    let m = VGpuMemory::new(64, 16, 3, 4);
    assert_eq!(m.shared_per_block.len(), 3);
    assert_eq!(m.locals_per_thread.len(), 3 * 4);
}
