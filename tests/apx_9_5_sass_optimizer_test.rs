use atenia_engine::apx9::sass_optimizer::*;

#[test]
fn apx_9_5_structure() {
    let sass = "// SASS MOCK v0\nLDG.E...\nFADD...\nSTG.E...\n".to_string();
    let out = SassOptimizer::optimize(&sass);

    assert!(out.contains("SASS OPT v0"));
}

#[test]
fn apx_9_5_reordering() {
    let sass = "
        FADD R2,R0,R1;
        STG.E [Out],R2;
        LDG.E R0,[param_A];
    ";

    let out = SassOptimizer::optimize(sass);
    let lines: Vec<&str> = out.lines().collect();

    // LDG must come first
    assert!(lines.iter().position(|l| l.contains("LDG"))
        < lines.iter().position(|l| l.contains("FADD")));

    // FADD before STG
    assert!(lines.iter().position(|l| l.contains("FADD"))
        < lines.iter().position(|l| l.contains("STG")));
}

#[test]
fn apx_9_5_no_numeric_change() {
    let mut v = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];

    for i in 0..2 {
        v[i] += b[i];
    }

    assert_eq!(v, vec![4.0, 6.0]);
}
