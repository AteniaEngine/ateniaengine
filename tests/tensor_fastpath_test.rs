use atenia_engine::tensor::ops::testing::{fast_add_calls, reset_op_counters, slow_add_calls};
use atenia_engine::tensor::tensor::{Device, DType, Layout, Tensor};

fn assign_logical_values(tensor: &mut Tensor, values: &[f32]) {
    assert_eq!(values.len(), tensor.num_elements());

    if tensor.shape.is_empty() {
        tensor.data[0] = values[0];
        return;
    }

    let mut index = vec![0usize; tensor.shape.len()];
    let mut value_iter = values.iter();
    loop {
        let offset = linear_offset(&index, &tensor.strides);
        tensor.data[offset] = *value_iter.next().unwrap();
        if !increment_index(&mut index, &tensor.shape) {
            break;
        }
    }
}

fn collect_logical_values(tensor: &Tensor) -> Vec<f32> {
    let mut result = vec![0.0; tensor.num_elements()];
    if tensor.shape.is_empty() {
        result[0] = tensor.data[0];
        return result;
    }

    let mut index = vec![0usize; tensor.shape.len()];
    let mut pos = 0;
    loop {
        let offset = linear_offset(&index, &tensor.strides);
        result[pos] = tensor.data[offset];
        pos += 1;
        if !increment_index(&mut index, &tensor.shape) {
            break;
        }
    }

    result
}

fn linear_offset(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides)
        .map(|(i, s)| i * s)
        .sum()
}

fn increment_index(index: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..index.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return true;
        }
        index[axis] = 0;
    }
    false
}

#[test]
fn fast_path_used_for_contiguous_tensors() {
    reset_op_counters();

    let shape = vec![4];
    let mut a = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let mut b = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    assign_logical_values(&mut a, &[1.0, 2.0, 3.0, 4.0]);
    assign_logical_values(&mut b, &[4.0, 3.0, 2.0, 1.0]);

    let result = a.add(&b);
    assert_eq!(fast_add_calls(), 1);
    assert_eq!(slow_add_calls(), 0);
    assert_eq!(collect_logical_values(&result), vec![5.0, 5.0, 5.0, 5.0]);
}

#[test]
fn slow_path_used_when_layouts_differ() {
    reset_op_counters();

    let shape = vec![1, 2, 2, 2];
    let mut a = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let mut b = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::ChannelsLast,
        DType::F32,
    );
    assign_logical_values(&mut a, &(1..=8).map(|v| v as f32).collect::<Vec<_>>());
    assign_logical_values(&mut b, &(8..16).map(|v| v as f32).collect::<Vec<_>>());

    let result = a.add(&b);
    assert_eq!(fast_add_calls(), 0);
    assert_eq!(slow_add_calls(), 1);
    let expected: Vec<f32> = collect_logical_values(&a)
        .into_iter()
        .zip(collect_logical_values(&b))
        .map(|(lhs, rhs)| lhs + rhs)
        .collect();
    assert_eq!(collect_logical_values(&result), expected);
}

#[test]
fn fast_and_slow_paths_produce_identical_results() {
    reset_op_counters();

    let shape = vec![1, 3, 2, 2];
    let values_a: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let values_b: Vec<f32> = (12..24).map(|v| v as f32).collect();

    let mut fast_a = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let mut fast_b = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    assign_logical_values(&mut fast_a, &values_a);
    assign_logical_values(&mut fast_b, &values_b);

    let mut slow_a = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::ChannelsLast,
        DType::F32,
    );
    let mut slow_b = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::ChannelsLast,
        DType::F32,
    );
    assign_logical_values(&mut slow_a, &values_a);
    assign_logical_values(&mut slow_b, &values_b);

    let fast_result = fast_a.mul(&fast_b);
    let slow_result = slow_a.mul(&slow_b);

    let expected: Vec<f32> = values_a.iter().zip(&values_b).map(|(a, b)| a * b).collect();
    assert_eq!(collect_logical_values(&fast_result), expected);
    assert_eq!(collect_logical_values(&slow_result), expected);
}
