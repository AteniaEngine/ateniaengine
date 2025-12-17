use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{Tensor, Device, DType, Layout};

fn assert_close(a: &Tensor, b: &Tensor, tol: f32) {
    assert_eq!(a.shape, b.shape, "shape mismatch: {:?} vs {:?}", a.shape, b.shape);
    assert_eq!(a.data.len(), b.data.len(), "len mismatch");
    for (i, (va, vb)) in a.data.iter().zip(b.data.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "mismatch at idx {}: a={} b={} diff={} tol={}",
            i,
            va,
            vb,
            diff,
            tol
        );
    }
}

fn build_self_attention_graph() -> (GraphBuilder, usize, usize, usize, usize) {
    let mut gb = GraphBuilder::new();
    let x = gb.input();
    let wq = gb.input();
    let wk = gb.input();
    let wv = gb.input();

    let q = gb.linear(x, wq, None);
    let k = gb.linear(x, wk, None);
    let v = gb.linear(x, wv, None);

    let k_t = gb.transpose_last_two(k);
    let qk = gb.matmul(q, k_t);
    let att = gb.softmax(qk);
    let out = gb.matmul(att, v);
    let _ = gb.output(out);

    (gb, x, wq, wk, wv)
}

fn run_mode(mode: &str) -> (Tensor, Tensor, Tensor, Tensor) {
    if mode == "naive" {
        // Baseline completamente sin fusiones APX 4.x: forzamos 2.5
        unsafe { std::env::set_var("ATENIA_APX_MODE", "2.5"); }
    } else {
        unsafe { std::env::set_var("ATENIA_APX_MODE", mode); }
    }

    let (gb, x_id, wq_id, wk_id, wv_id) = build_self_attention_graph();
    let mut graph = gb.build();

    let m = 2usize;
    let d = 4usize;

    let x = Tensor::with_layout(vec![m, d], 0.5, Device::CPU, Layout::Contiguous, DType::F32);
    let wq = Tensor::with_layout(vec![d, d], 0.1, Device::CPU, Layout::Contiguous, DType::F32);
    let wk = Tensor::with_layout(vec![d, d], -0.2, Device::CPU, Layout::Contiguous, DType::F32);
    let wv = Tensor::with_layout(vec![d, d], 0.3, Device::CPU, Layout::Contiguous, DType::F32);

    let inputs = vec![x.clone(), wq.clone(), wk.clone(), wv.clone()];

    let out_id = graph.last_output_id();
    let _outs = graph.execute(inputs);
    graph.backward(out_id);

    let x_t  = graph.nodes[x_id].output.as_ref().expect("x output missing").clone();
    let wq_t = graph.nodes[wq_id].output.as_ref().expect("wq output missing").clone();
    let wk_t = graph.nodes[wk_id].output.as_ref().expect("wk output missing").clone();
    let wv_t = graph.nodes[wv_id].output.as_ref().expect("wv output missing").clone();

    let make_grad = |node_id: usize, proto: &Tensor| -> Tensor {
        let data = graph.nodes[node_id]
            .output
            .as_ref()
            .and_then(|t| t.grad.as_ref())
            .cloned()
            .unwrap_or_else(|| vec![0.0; proto.data.len()]);
        Tensor {
            shape: proto.shape.clone(),
            data,
            device: proto.device,
            dtype: proto.dtype,
            layout: proto.layout,
            strides: proto.strides.clone(),
            grad: None,
            gpu: None,
            persistence: None,
            op: None,
        }
    };

    let dx  = make_grad(x_id,  &x_t);
    let dwq = make_grad(wq_id, &wq_t);
    let dwk = make_grad(wk_id, &wk_t);
    let dwv = make_grad(wv_id, &wv_t);

    (dx, dwq, dwk, dwv)
}

#[test]
fn self_attention_backward_4_18_matches_naive() {
    let (dx_naive, dwq_naive, dwk_naive, dwv_naive) = run_mode("naive");
    let (dx_4_18, dwq_4_18, dwk_4_18, dwv_4_18) = run_mode("4.18");

    let tol = 1e-5;
    assert_close(&dx_naive,  &dx_4_18,  tol);
    assert_close(&dwq_naive, &dwq_4_18, tol);
    assert_close(&dwk_naive, &dwk_4_18, tol);
    assert_close(&dwv_naive, &dwv_4_18, tol);
}
