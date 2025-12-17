use atenia_engine::apx6_8::BlockSizePredictor;

#[test]
fn selector_uses_predicted_block_size() {
    let mut pred = BlockSizePredictor::new();
    pred.record_result(64, 64, 32, 50000);
    pred.record_result(128, 64, 32, 40000); // mejor
    pred.record_result(64, 128, 32, 70000);

    let best = pred.best_block().unwrap();
    assert_eq!(best, (128, 64, 32));
}
