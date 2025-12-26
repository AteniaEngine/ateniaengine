use atenia_engine::v13::compression::{rle_compress, rle_decompress};
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{
    CompressionKind, CompressionMeta, MemoryTier, MoveError, StorageBacking,
};
use atenia_engine::v13::ssd_cache::SsdCache;

use std::fs;

fn make_bytes() -> Vec<u8> {
    // Include repetitions so that RLE actually changes the representation.
    let mut v = Vec::new();
    v.extend(std::iter::repeat(1u8).take(8));
    v.extend(std::iter::repeat(2u8).take(4));
    v.extend(std::iter::repeat(3u8).take(2));
    v
}

#[test]
fn rle_roundtrip_integrity() {
    let input = make_bytes();

    let (compressed, meta) = rle_compress(&input);
    assert!(compressed.len() <= input.len() * 2);
    assert_eq!(meta.kind, CompressionKind::Rle);
    assert_eq!(meta.original_bytes, input.len() as u64);

    let decompressed = match rle_decompress(&compressed, &meta) {
        Ok(v) => v,
        Err(e) => panic!("rle_decompress should succeed: {:?}", e),
    };

    assert_eq!(decompressed, input);
}

#[test]
fn ssd_roundtrip_with_compression_meta_none() {
    let cache_dir = "./.atenia_cache_test_compression_none";
    let _ = fs::remove_dir_all(cache_dir);

    let mut mgr = HybridMemoryManager::new(cache_dir);
    let data = make_bytes();
    let id = "tensor_compression_none";

    if let Err(e) = mgr.register_tensor_with_data(id, data.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    let snapshot = atenia_engine::v13::memory_types::MemorySnapshot {
        vram: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ram: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ssd: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
    };

    // RAM -> SSD (default policy: CompressionKind::None).
    let plan_to_ssd = match mgr.plan_move(id, MemoryTier::Ssd, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to SSD should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_ssd) {
        panic!("apply_move to SSD should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ssd));
    match mgr.backing_for_test(id) {
        Some(StorageBacking::SsdFile { path, compression }) => {
            assert!(std::path::Path::new(path).exists());
            match compression {
                Some(meta) => {
                    assert_eq!(meta.kind, CompressionKind::None);
                    assert_eq!(meta.original_bytes, data.len() as u64);
                }
                None => panic!("Expected Some(CompressionMeta) for SSD backing"),
            }
        }
        other => panic!("Expected SsdFile backing, got {:?}", other),
    }

    // SSD -> RAM using compression metadata.
    let plan_to_ram = match mgr.plan_move(id, MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to RAM should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_ram) {
        panic!("apply_move to RAM should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ram));
    match mgr.backing_for_test(id) {
        Some(StorageBacking::Ram(bytes)) => {
            assert_eq!(bytes, &data);
        }
        other => panic!("Expected RAM backing after SSD roundtrip, got {:?}", other),
    }

    let _ = fs::remove_dir_all(cache_dir);
}

#[test]
fn ssd_roundtrip_with_rle_compression() {
    let cache_dir = "./.atenia_cache_test_compression_rle";
    let _ = fs::remove_dir_all(cache_dir);

    let cache = SsdCache::new(cache_dir);
    if let Err(e) = cache.ensure_dir() {
        panic!("ensure_dir should succeed: {:?}", e);
    }

    let data = make_bytes();
    let path = cache.blob_path("tensor_rle");

    // Write using RLE compression directly through SsdCache.
    let meta = match cache.write_blob(&path, &data, CompressionKind::Rle) {
        Ok(m) => m,
        Err(e) => panic!("write_blob with RLE should succeed: {:?}", e),
    };

    assert_eq!(meta.kind, CompressionKind::Rle);
    assert_eq!(meta.original_bytes, data.len() as u64);
    assert!(std::path::Path::new(&path).exists());

    // Read back using metadata (should transparently decompress).
    let roundtrip = match cache.read_blob_with_meta(&path, &meta) {
        Ok(v) => v,
        Err(e) => panic!("read_blob_with_meta should succeed: {:?}", e),
    };

    assert_eq!(roundtrip, data);

    let _ = fs::remove_dir_all(cache_dir);
}

#[test]
fn invalid_rle_stream_returns_error() {
    // Malformed RLE: odd length and zero count cases.
    let meta = CompressionMeta {
        kind: CompressionKind::Rle,
        original_bytes: 1,
    };

    // Odd length input should fail.
    let bytes_odd = vec![1u8, 42u8, 3u8];
    match rle_decompress(&bytes_odd, &meta) {
        Err(MoveError::Unsupported(msg)) => {
            assert!(msg.contains("Invalid RLE stream"));
        }
        other => panic!("Expected Unsupported for odd-length RLE, got {:?}", other),
    }

    // Zero count should fail.
    let bytes_zero_count = vec![0u8, 10u8];
    match rle_decompress(&bytes_zero_count, &meta) {
        Err(MoveError::Unsupported(msg)) => {
            assert!(msg.contains("Invalid RLE stream"));
        }
        other => panic!("Expected Unsupported for zero-count RLE, got {:?}", other),
    }
}
