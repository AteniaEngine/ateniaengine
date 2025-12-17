use crate::tensor::Tensor;
use crate::amg::grad_store::GradStore;
use crate::apx3::grad_pipeline::chunk_size;

pub fn fused_linear_backward(
    store: &GradStore,
    input: &Tensor,
    weight: &Tensor,
    out_grad: &Tensor,
    input_grad_id: usize,
    weight_grad_id: usize,
) {
    assert!(input.shape.len() == 2, "fused_linear_backward expects 2D input");
    assert!(weight.shape.len() == 2, "fused_linear_backward expects 2D weight");

    let batch = input.shape[0];
    let in_dim = input.shape[1];
    let out_dim = weight.shape[1];

    assert_eq!(weight.shape[0], in_dim, "weight rows must match in_dim");
    assert_eq!(out_grad.data.len(), batch * out_dim, "out_grad shape mismatch");

    let mut d_w = vec![0.0f32; in_dim * out_dim];
    let mut d_x = vec![0.0f32; batch * in_dim];

    let chunk = chunk_size(batch);

    for start in (0..batch).step_by(chunk) {
        let end = (start + chunk).min(batch);

        for i in start..end {
            let x = &input.data[i * in_dim..(i + 1) * in_dim];
            let dy = &out_grad.data[i * out_dim..(i + 1) * out_dim];

            // dW += x outer dy  (layout [in_dim, out_dim])
            for o in 0..out_dim {
                for ii in 0..in_dim {
                    d_w[ii * out_dim + o] += x[ii] * dy[o];
                }
            }

            // dX = dy * W^T
            for ii in 0..in_dim {
                let mut acc = 0.0f32;
                for o in 0..out_dim {
                    acc += dy[o] * weight.data[ii * out_dim + o];
                }
                d_x[i * in_dim + ii] = acc;
            }
        }
    }

    // Accumulate into GradStore
    inplace_add_and_add(store, input_grad_id, &d_x);
    inplace_add_and_add(store, weight_grad_id, &d_w);
}

fn inplace_add_and_add(store: &GradStore, id: usize, grad: &[f32]) {
    // GradStore::add already accumulates, but we keep this helper
    // in case we later want to use more advanced memory flows.
    store.add(id, grad);
}
