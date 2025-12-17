use atenia_engine::apx6_6_auto_tiling::{AutoTilingSelector, KernelKind};

#[test]
fn selector_chooses_expected_kernel() {
    assert_eq!(
        AutoTilingSelector::choose_kernel(64, 64, 64),
        KernelKind::Baseline38
    );

    assert_eq!(
        AutoTilingSelector::choose_kernel(256, 256, 256),
        KernelKind::Micro64
    );
}

#[test]
fn selector_chooses_expected_tiles() {
    let tiles = AutoTilingSelector::choose_tile_sizes(1024, 1024, 1024);
    assert_eq!(tiles.bm, 64);
    assert_eq!(tiles.bn, 64);
    assert_eq!(tiles.bk, 64);
}
