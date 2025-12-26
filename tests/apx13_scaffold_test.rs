use atenia_engine::v13::types::BackendKind;

#[test]
fn apx13_scaffold_compiles() {
    let b = BackendKind::Unknown;
    match b {
        BackendKind::Unknown => {}
        _ => {}
    }
}
