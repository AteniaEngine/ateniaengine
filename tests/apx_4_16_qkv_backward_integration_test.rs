use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::tensor::{Tensor, Device, DType, Layout};

fn build_qkv_sum_graph_with_bias() -> (Graph, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize) {
    let mut gb = GraphBuilder::new();
    let x  = gb.input();      // 0
    let wq = gb.input();      // 1
    let wk = gb.input();      // 2
    let wv = gb.input();      // 3
    let bq = gb.input();      // 4
    let bk = gb.input();      // 5
    let bv = gb.input();      // 6

    let q = gb.linear(x, wq, Some(bq));
    let k = gb.linear(x, wk, Some(bk));
    let v = gb.linear(x, wv, Some(bv));

    let sum1 = gb.add(q, k);
    let sum2 = gb.add(sum1, v);
    let _out = gb.output(sum2);

    let g = gb.build();

    (g, x, wq, wk, wv, bq, bk, bv, q, k, v)
}

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

struct Grads {
    dx: Tensor,
    dwq: Tensor,
    dwk: Tensor,
    dwv: Tensor,
    dbq: Tensor,
    dbk: Tensor,
    dbv: Tensor,
}

fn run_mode(mode: &str) -> Grads {
    // Modo "naive": sin APX_4.13/4.14/4.16 activos, para usar backward baseline.
    if mode == "naive" {
        unsafe { std::env::remove_var("ATENIA_APX_MODE"); }
    } else {
        unsafe { std::env::set_var("ATENIA_APX_MODE", mode); }
    }

    let (mut graph, x_id, wq_id, wk_id, wv_id, bq_id, bk_id, bv_id, _q_id, _k_id, _v_id) =
        build_qkv_sum_graph_with_bias();

    // Small, deterministic dimensions.
    let m = 2usize;
    let k = 4usize;
    let n = 3usize;

    let x = Tensor::with_layout(vec![m, k], 0.5, Device::CPU, Layout::Contiguous, DType::F32);
    let wq = Tensor::with_layout(vec![k, n], 0.1, Device::CPU, Layout::Contiguous, DType::F32);
    let wk = Tensor::with_layout(vec![k, n], -0.2, Device::CPU, Layout::Contiguous, DType::F32);
    let wv = Tensor::with_layout(vec![k, n], 0.3, Device::CPU, Layout::Contiguous, DType::F32);

    let bq = Tensor::with_layout(vec![n], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    let bk = Tensor::with_layout(vec![n], 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    let bv = Tensor::with_layout(vec![n], 0.0, Device::CPU, Layout::Contiguous, DType::F32);

    let inputs = vec![x.clone(), wq.clone(), wk.clone(), wv.clone(), bq.clone(), bk.clone(), bv.clone()];

    let out_id = graph.last_output_id();
    let _outs = graph.execute(inputs);
    graph.backward(out_id);

    let x_t  = graph.nodes[x_id].output.as_ref().expect("x output missing").clone();
    let wq_t = graph.nodes[wq_id].output.as_ref().expect("wq output missing").clone();
    let wk_t = graph.nodes[wk_id].output.as_ref().expect("wk output missing").clone();
    let wv_t = graph.nodes[wv_id].output.as_ref().expect("wv output missing").clone();
    let bq_t = graph.nodes[bq_id].output.as_ref().expect("bq output missing").clone();
    let bk_t = graph.nodes[bk_id].output.as_ref().expect("bk output missing").clone();
    let bv_t = graph.nodes[bv_id].output.as_ref().expect("bv output missing").clone();

    let make_grad = |node_id: usize, proto: &Tensor| -> Tensor {
        // 1) Try to read grad directly from the output tensor.
        let from_tensor = graph.nodes[node_id]
            .output
            .as_ref()
            .and_then(|t| t.grad.as_ref())
            .cloned();

        // 2) If there is no grad in the tensor, try GradStore.
        let data = if let Some(g) = from_tensor {
            g
        } else {
            let g = graph.grad_store.get(node_id);
            if g.is_empty() {
                // 3) As a last resort, return a zero gradient of the same size.
                vec![0.0; proto.data.len()]
            } else {
                g
            }
        };

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
    let dbq = make_grad(bq_id, &bq_t);
    let dbk = make_grad(bk_id, &bk_t);
    let dbv = make_grad(bv_id, &bv_t);

    Grads { dx, dwq, dwk, dwv, dbq, dbk, dbv }
}

#[test]
fn test_qkv_backward_4_14_vs_4_16_match() {
    // Baseline naive: sin fusiones APX 4.x
    let g_naive = run_mode("naive");
    // 4.16: forward QKV fusionado + backward fusionado real
    let g_4_16 = run_mode("4.16");

    let tol = 1e-6;
    assert_close(&g_naive.dx,  &g_4_16.dx,  tol);
    assert_close(&g_naive.dwq, &g_4_16.dwq, tol);
    assert_close(&g_naive.dwk, &g_4_16.dwk, tol);
    assert_close(&g_naive.dwv, &g_4_16.dwv, tol);
    assert_close(&g_naive.dbq, &g_4_16.dbq, tol);
    assert_close(&g_naive.dbk, &g_4_16.dbk, tol);
    assert_close(&g_naive.dbv, &g_4_16.dbv, tol);
}
