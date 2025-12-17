use atenia_engine::amm::memory_manager::{ManagedTensor, MemoryManager};
use atenia_engine::tensor::{Device, DType, Layout, Tensor};

fn make_tensor(len: usize, value: f32) -> Tensor {
    let mut t = Tensor::with_layout(
        vec![len],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    for v in t.data.iter_mut() {
        *v = value;
    }
    t
}

#[test]
fn estimate_total_bytes_matches_sum() {
    let mut mm = MemoryManager::new(1024 * 1024, 0, "tmp_offload_mm".to_string());

    let t1 = make_tensor(10, 1.0);
    let t2 = make_tensor(20, 2.0);

    let mts = vec![
        ManagedTensor {
            tensor: Some(t1.clone()),
            offload: None,
        },
        ManagedTensor {
            tensor: Some(t2.clone()),
            offload: None,
        },
    ];

    let total = mm.estimate_total_bytes(&mts);

    let expected = t1.estimated_bytes() + t2.estimated_bytes();
    assert_eq!(total, expected);
}

#[test]
fn enforce_limit_offloads_largest_tensor() {
    let limit = 10 * 1024; // 10 KB
    let margin = 0;

    let mut mm = MemoryManager::new(limit, margin, "tmp_offload_mm2".to_string());

    let small = make_tensor(16, 1.0);
    let large = make_tensor(4096, 2.0);

    let mut mts = vec![
        ManagedTensor {
            tensor: Some(small),
            offload: None,
        },
        ManagedTensor {
            tensor: Some(large),
            offload: None,
        },
    ];

    let offloaded_idx = mm.enforce_limit(&mut mts);

    assert!(offloaded_idx.is_some());
    let idx = offloaded_idx.unwrap();

    assert!(mts[idx].tensor.is_none());
    assert!(mts[idx].offload.is_some());
}

#[test]
fn load_from_disk_restores_tensor() {
    let limit = 64; // force offloading even for modest tensors
    let margin = 0;

    let mut mm = MemoryManager::new(limit, margin, "tmp_offload_mm3".to_string());

    let t = make_tensor(32, 3.0);

    let mut mts = vec![ManagedTensor {
        tensor: Some(t.clone()),
        offload: None,
    }];

    let idx = mm
        .enforce_limit(&mut mts)
        .expect("should offload something");

    assert!(mts[idx].tensor.is_none());
    assert!(mts[idx].offload.is_some());

    let shape = vec![32];
    mm.load_from_disk(&mut mts[idx], shape);

    assert!(mts[idx].tensor.is_some());
    let restored = mts[idx].tensor.as_ref().unwrap();

    assert_eq!(restored.data.len(), t.data.len());
    for (a, b) in restored.data.iter().zip(t.data.iter()) {
        assert_eq!(*a, *b);
    }
}
