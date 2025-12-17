use atenia_engine::tensor::tensor::{Device, DType, Layout, Tensor};

#[test]
fn contiguous_strides_are_correct() {
    let shape = vec![2, 3, 4];
    let strides = Tensor::compute_strides(&shape, &Layout::Contiguous);
    assert_eq!(strides, vec![12, 4, 1]);
}

#[test]
fn channels_layouts_differ() {
    let shape = vec![1, 3, 32, 32];
    let nchw = Tensor::compute_strides(&shape, &Layout::ChannelsFirst);
    let nhwc = Tensor::compute_strides(&shape, &Layout::ChannelsLast);
    assert_ne!(nchw, nhwc);
    assert_eq!(nchw, vec![3072, 1024, 32, 1]); // NCHW row-major
    assert_eq!(nhwc, vec![3072, 1, 96, 3]);    // NHWC row-major
}

#[test]
fn add_respects_layout() {
    let shape = vec![1, 3, 2, 2];
    let mut a = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::ChannelsLast,
        DType::F32,
    );
    let mut b = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::ChannelsLast,
        DType::F32,
    );

    a.data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    b.data = vec![12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let result = a.add(&b);
    assert_eq!(result.layout, Layout::ChannelsLast);
    assert_eq!(result.strides, Tensor::compute_strides(&shape, &Layout::ChannelsLast));
    assert_eq!(result.data[0], 13.0);
    assert_eq!(result.data[11], 13.0);
}

#[test]
fn with_layout_constructor_sets_fields() {
    let shape = vec![1, 3, 4, 4];
    let tensor = Tensor::with_layout(
        shape.clone(),
        2.0,
        Device::GPU,
        Layout::ChannelsFirst,
        DType::F32,
    );
    assert_eq!(tensor.layout, Layout::ChannelsFirst);
    assert_eq!(tensor.strides, Tensor::compute_strides(&shape, &Layout::ChannelsFirst));
    assert_eq!(tensor.device, Device::GPU);
    assert!(tensor.data.iter().all(|&v| (v - 2.0).abs() < f32::EPSILON));
}
