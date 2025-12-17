use atenia_engine::apx7_2::pgl::{decide_pgl, PGLStrategy};

#[test]
fn apx_7_2_small_mat_chooses_seq() {
    let d = decide_pgl(32, 32, 32, 24);
    assert!(matches!(d.strategy, PGLStrategy::Seq));
}

#[test]
fn apx_7_2_medium_mat_chooses_pex() {
    let d = decide_pgl(512, 512, 512, 4);
    assert!(matches!(d.strategy, PGLStrategy::Pex));
}

#[test]
fn apx_7_2_large_mat_chooses_ws() {
    let d = decide_pgl(1024, 1024, 1024, 24);
    assert!(matches!(d.strategy, PGLStrategy::WorkStealing));
}
