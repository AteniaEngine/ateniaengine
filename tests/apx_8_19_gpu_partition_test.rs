use atenia_engine::apx8::gpu_partition::*;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_19_small_vector_no_split() {
    let p = suggest_partition(&[512]);
    match p.policy {
        PartitionPolicy::None => {}
        _ => panic!("expected no partition"),
    }
}

#[test]
fn apx_8_19_large_vector_split() {
    let p = suggest_partition(&[50000]);
    match p.policy {
        PartitionPolicy::Split1D { chunks } => assert_eq!(chunks, 2),
        _ => panic!("expected 1D split"),
    }
}

#[test]
fn apx_8_19_big_mat_split_2d() {
    let p = suggest_partition(&[2048, 2048]);
    match p.policy {
        PartitionPolicy::Split2D { rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 2);
        }
        _ => panic!("expected 2D split"),
    }
}

#[test]
fn apx_8_19_small_mat_no_split() {
    let p = suggest_partition(&[32, 32]);
    assert!(matches!(p.policy, PartitionPolicy::None));
}

#[test]
fn apx_8_19_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let c = a.add(&b);
    assert_eq!(c.data[0], 2.0);
}
