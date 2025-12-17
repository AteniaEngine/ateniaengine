use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{Tensor, Device, DType, Layout};

#[test]
fn test_fused_linear_silu_pipeline() {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w_id = gb.input();
    let b_id = gb.input();

    let lin = gb.linear(x_id, w_id, Some(b_id));
    let act = gb.silu(lin);
    gb.output(act);

    let mut g = gb.build();
    g.validate().unwrap();

    let m = 4usize;
    let k = 8usize;
    let n = 4usize;

    let x = Tensor::ones(vec![m, k], Device::CPU, DType::F32);
    let w = Tensor::ones(vec![k, n], Device::CPU, DType::F32);
    let b = Tensor::with_layout(vec![n], 0.0, Device::CPU, Layout::Contiguous, DType::F32);

    let out = g.execute(vec![x, w, b]);

    assert_eq!(out.len(), 1);
    assert!(out[0].data.len() > 0);
}
