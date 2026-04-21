use crate::tensor::{DType, Device, Layout, Tensor};

/// Split a tensor into multiple tensors with at most `max_elements` elements each,
/// preserving 1D semantics. For now we restrict to 1D data (shape [N]).
pub fn chunk_tensor(t: &Tensor, max_elements: usize) -> Vec<Tensor> {
    assert!(
        t.shape.len() == 1,
        "chunk_tensor currently supports only 1D tensors"
    );

    let total = t.numel();
    if max_elements == 0 || total == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < total {
        let end = usize::min(start + max_elements, total);
        let slice = &t.as_cpu_slice()[start..end];

        let mut data = Vec::with_capacity(slice.len());
        data.extend_from_slice(slice);

        let mut chunk = Tensor::new_cpu_with_layout(
            vec![(end - start)],
            data,
            t.device,
            t.dtype,
            t.layout.clone(),
        );
        chunk.strides = t.strides.clone();

        chunks.push(chunk);
        start = end;
    }

    chunks
}

/// Merge 1D chunks back into a single tensor with the original shape.
pub fn merge_chunks(chunks: Vec<Tensor>, original_shape: Vec<usize>) -> Tensor {
    if chunks.is_empty() {
        return Tensor::with_layout(
            original_shape,
            0.0,
            Device::CPU,
            Layout::Contiguous,
            DType::F32,
        );
    }

    let device = chunks[0].device;
    let dtype = chunks[0].dtype;
    let layout = chunks[0].layout.clone();
    let strides = chunks[0].strides.clone();

    let total_len: usize = chunks.iter().map(|c| c.numel()).sum();
    let mut data = Vec::with_capacity(total_len);

    for c in chunks {
        data.extend_from_slice(c.as_cpu_slice());
    }

    let mut out = Tensor::new_cpu_with_layout(
        original_shape,
        data,
        device,
        dtype,
        layout,
    );
    out.strides = strides;
    out
}
