use std::fs::{self, OpenOptions};
use std::io::Write;

use atenia_engine::v13::persistent_cache::{
    CacheError, CacheKind, PersistentHybridCache,
};

#[test]
fn put_and_get_roundtrip_tensor() {
    let root = "./.atenia_cache_test_persistent_roundtrip";
    let _ = fs::remove_dir_all(root);

    let cache = PersistentHybridCache::new(root);

    let bytes = vec![1u8, 2, 3];

    match cache.put_blob(CacheKind::Tensor, "w1", &bytes, 1234, false) {
        Ok(()) => {}
        Err(e) => panic!("put_blob should succeed: {:?}", e),
    }

    assert!(cache.exists(CacheKind::Tensor, "w1"));

    let loaded = match cache.get_blob(CacheKind::Tensor, "w1") {
        Ok(b) => b,
        Err(e) => panic!("get_blob should succeed: {:?}", e),
    };

    assert_eq!(loaded, bytes);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn put_rejects_overwrite_by_default() {
    let root = "./.atenia_cache_test_persistent_overwrite";
    let _ = fs::remove_dir_all(root);

    let cache = PersistentHybridCache::new(root);

    let bytes1 = vec![10u8];
    let bytes2 = vec![20u8];

    match cache.put_blob(CacheKind::Tensor, "k1", &bytes1, 1, false) {
        Ok(()) => {}
        Err(e) => panic!("first put_blob should succeed: {:?}", e),
    }

    match cache.put_blob(CacheKind::Tensor, "k1", &bytes2, 2, false) {
        Ok(()) => panic!("second put_blob should have failed with AlreadyExists"),
        Err(CacheError::AlreadyExists) => {}
        Err(e) => panic!("second put_blob returned wrong error: {:?}", e),
    }
}

#[test]
fn get_detects_corruption_by_checksum() {
    let root = "./.atenia_cache_test_persistent_corrupt";
    let _ = fs::remove_dir_all(root);
    let cache = PersistentHybridCache::new(root);

    let bytes = vec![5u8, 6, 7];

    if let Err(e) = cache.put_blob(CacheKind::Tensor, "c1", &bytes, 42, false) {
        panic!("put_blob should succeed: {:?}", e);
    }

    // Manually corrupt the .bin file by appending a byte.
    let bin_path = std::path::Path::new(root)
        .join("tensor")
        .join("c1.bin");

    let mut file = match OpenOptions::new().append(true).open(&bin_path) {
        Ok(f) => f,
        Err(e) => panic!("failed to open bin for corruption: {:?}", e),
    };

    if let Err(e) = file.write_all(&[0xFFu8]) {
        panic!("failed to append corruption byte: {:?}", e);
    }

    match cache.get_blob(CacheKind::Tensor, "c1") {
        Ok(_) => panic!("get_blob should have failed with Corrupt"),
        Err(CacheError::Corrupt(msg)) => {
            assert!(msg.to_lowercase().contains("mismatch"));
        }
        Err(e) => panic!("get_blob returned wrong error: {:?}", e),
    }
}

#[test]
fn separate_kinds_use_separate_namespaces() {
    let root = "./.atenia_cache_test_persistent_kinds";
    let _ = fs::remove_dir_all(root);
    let cache = PersistentHybridCache::new(root);

    let t_bytes = vec![1u8, 2, 3];
    let g_bytes = vec![9u8, 8, 7];

    if let Err(e) = cache.put_blob(CacheKind::Tensor, "k", &t_bytes, 10, false) {
        panic!("put_blob tensor should succeed: {:?}", e);
    }

    if let Err(e) = cache.put_blob(CacheKind::Gradient, "k", &g_bytes, 20, false) {
        panic!("put_blob gradient should succeed: {:?}", e);
    }

    assert!(cache.exists(CacheKind::Tensor, "k"));
    assert!(cache.exists(CacheKind::Gradient, "k"));

    let loaded_t = match cache.get_blob(CacheKind::Tensor, "k") {
        Ok(b) => b,
        Err(e) => panic!("get_blob tensor should succeed: {:?}", e),
    };

    let loaded_g = match cache.get_blob(CacheKind::Gradient, "k") {
        Ok(b) => b,
        Err(e) => panic!("get_blob gradient should succeed: {:?}", e),
    };

    assert_eq!(loaded_t, t_bytes);
    assert_eq!(loaded_g, g_bytes);
}
