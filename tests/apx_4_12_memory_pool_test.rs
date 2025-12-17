use atenia_engine::apx4_12::*;

#[test]
fn test_pool_alloc_free() {
    init_pool(1024, 2);
    let a = pool_alloc();
    let b = pool_alloc();
    pool_free(a);
    pool_free(b);
    assert!(true);
}
