use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::activations as nn_act;
use atenia_engine::nn::linear as nn_linear;
use atenia_engine::nn::normalization as nn_norm;
use atenia_engine::nn::softmax as nn_softmax;
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

fn make_tensor_2d(shape: (usize, usize), fill_fn: impl Fn(usize, usize) -> f32) -> Tensor {
    let (rows, cols) = shape;
    let mut t = Tensor::with_layout(
        vec![rows, cols],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for i in 0..rows {
        for j in 0..cols {
            t.data[i * cols + j] = fill_fn(i, j);
        }
    }
    t
}

#[test]
fn graph_linear_matches_direct_linear() {
    let mut gb = GraphBuilder::new();

    let x_id = gb.input();
    let w_id = gb.input();
    let b_id = gb.input();

    let lin_id = gb.linear(x_id, w_id, Some(b_id));
    let _out_id = gb.output(lin_id);

    let mut g = gb.build();

    let batch = 2;
    let in_features = 3;
    let out_features = 4;

    let x = make_tensor_2d((batch, in_features), |i, j| (i * in_features + j) as f32);
    let w = make_tensor_2d((in_features, out_features), |i, j| ((i + j) as f32) * 0.5);

    let mut b = Tensor::with_layout(
        vec![out_features],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for j in 0..out_features {
        b.data[j] = j as f32;
    }

    let direct = nn_linear::linear(&x, &w, Some(&b));

    let graph_out = g.execute(vec![x.clone(), w.clone(), b.clone()]);
    assert_eq!(graph_out.len(), 1);

    let y = &graph_out[0];

    assert_eq!(y.shape, direct.shape);
    assert_eq!(y.data.len(), direct.data.len());

    for (a, b) in y.data.iter().zip(direct.data.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn graph_rms_silu_softmax_pipeline_is_valid() {
    let mut gb = GraphBuilder::new();

    let x_id = gb.input();
    let n_id = gb.rms_norm(x_id);
    let a_id = gb.silu(n_id);
    let s_id = gb.softmax(a_id);
    let _out_id = gb.output(s_id);

    let mut g = gb.build();

    let rows = 3;
    let cols = 5;
    let x = make_tensor_2d((rows, cols), |i, j| (i * cols + j) as f32 - 5.0);

    let graph_out = g.execute(vec![x.clone()]);
    assert_eq!(graph_out.len(), 1);

    let y = &graph_out[0];

    assert_eq!(y.shape, vec![rows, cols]);

    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let row = &y.data[start..end];

        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {} sum = {}", r, sum);

        for p in row {
            assert!(*p >= 0.0);
            assert!(*p <= 1.0 + 1e-5);
        }
    }

    // Compare against direct pipeline for sanity.
    let rms = nn_norm::rms_norm(&x, 1e-5);
    let silu = nn_act::silu(&rms);
    let direct_softmax = nn_softmax::softmax_last_dim(&silu);

    for (a, b) in y.data.iter().zip(direct_softmax.data.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}
