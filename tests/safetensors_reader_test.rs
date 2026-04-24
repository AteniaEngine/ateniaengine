//! Integration tests for `src/v17/loader/safetensors_reader.rs` (M4-a).
//!
//! Uses the `safetensors::serialize` writer to produce synthetic
//! fixture bytes in-memory, then feeds them back through
//! `SafetensorsReader::from_bytes`. This avoids committing binary
//! fixtures and also confirms that Atenia's reader interoperates
//! with the reference HuggingFace writer.

use std::collections::HashMap;

use atenia_engine::tensor::DType;
use atenia_engine::v17::loader::loader_errors::LoaderError;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

/// Helper: build a raw byte payload from an `&[f32]` in
/// little-endian form, matching the on-disk representation
/// safetensors uses for F32 tensors.
fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
fn parses_well_formed_single_tensor_f32() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2usize, 3];
    let bytes = f32_slice_to_le_bytes(&values);
    let view = TensorView::new(StDtype::F32, shape.clone(), &bytes).unwrap();

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert("weight".to_string(), view);
    let serialized = safetensors::serialize(&tensors, &None).unwrap();

    let reader = SafetensorsReader::from_bytes(serialized).expect("reader opens");

    assert_eq!(reader.len(), 1);

    let entry = reader.get("weight").expect("weight present");
    assert_eq!(entry.name, "weight");
    assert_eq!(entry.shape, &[2, 3]);
    assert_eq!(entry.dtype, DType::F32);
    assert_eq!(entry.raw_bytes.len(), 24); // 6 f32 = 24 bytes
}

#[test]
fn to_vec_f32_roundtrips_values_bit_exact() {
    let values: Vec<f32> = vec![
        1.0,
        -0.5,
        std::f32::consts::PI,
        std::f32::consts::E,
        0.0,
        -1.0e-7,
        1.234_567_8,
        -9.876_543_2,
    ];
    let shape = vec![8usize];
    let bytes = f32_slice_to_le_bytes(&values);
    let view = TensorView::new(StDtype::F32, shape.clone(), &bytes).unwrap();

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert("probe".to_string(), view);
    let serialized = safetensors::serialize(&tensors, &None).unwrap();

    let reader = SafetensorsReader::from_bytes(serialized).unwrap();
    let entry = reader.get("probe").unwrap();
    let decoded = entry.to_vec_f32().expect("F32 decode must succeed");

    assert_eq!(decoded.len(), values.len());
    for (i, (a, b)) in decoded.iter().zip(values.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {}: decoded {} bits={:08x}, expected {} bits={:08x}",
            i,
            a,
            a.to_bits(),
            b,
            b.to_bits()
        );
    }
}

#[test]
fn bf16_tensor_reports_unsupported_dtype_on_conversion() {
    // safetensors BF16: 2 bytes per element. Construct 4 dummy BF16
    // values (we do not care about their numeric meaning — we only
    // assert on the error path for to_vec_f32).
    let bf16_bytes: Vec<u8> = vec![0x3f, 0x80, 0xbf, 0x00, 0x40, 0x49, 0x00, 0x00];
    let shape = vec![4usize];
    let view = TensorView::new(StDtype::BF16, shape, &bf16_bytes).unwrap();

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert("bf16_tensor".to_string(), view);
    let serialized = safetensors::serialize(&tensors, &None).unwrap();

    let reader = SafetensorsReader::from_bytes(serialized).unwrap();
    let entry = reader.get("bf16_tensor").unwrap();
    assert_eq!(entry.dtype, DType::BF16);

    let err = entry.to_vec_f32().expect_err("BF16 decode must error");
    match err {
        LoaderError::UnsupportedDType(msg) => {
            assert!(
                msg.contains("BF16") && msg.contains("M4-d"),
                "error message should reference BF16 and M4-d: got {}",
                msg
            );
        }
        other => panic!("expected UnsupportedDType, got {:?}", other),
    }
}

#[test]
fn malformed_bytes_short_of_header_prefix_errors() {
    let too_short: Vec<u8> = vec![0x01, 0x02, 0x03];
    let err = SafetensorsReader::from_bytes(too_short)
        .expect_err("3-byte buffer must fail the 8-byte prefix check");
    match err {
        LoaderError::InvalidFormat(_) => {}
        other => panic!("expected InvalidFormat, got {:?}", other),
    }
}

#[test]
fn malformed_bytes_invalid_header_json_errors() {
    // 8-byte prefix says the header is 20 bytes long, followed by 20
    // bytes of garbage that is not valid JSON.
    let header_len: u64 = 20;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&[0xff; 20]);
    let err = SafetensorsReader::from_bytes(bytes)
        .expect_err("garbage header must fail JSON parse");
    match err {
        LoaderError::InvalidFormat(msg) => {
            assert!(
                !msg.is_empty(),
                "InvalidFormat error should carry a non-empty message"
            );
        }
        other => panic!("expected InvalidFormat, got {:?}", other),
    }
}

#[test]
fn get_returns_some_for_existing_and_none_for_missing() {
    // Build a safetensors with three tensors.
    let a = vec![1.0f32, 2.0];
    let b = vec![3.0f32, 4.0, 5.0];
    let c = vec![6.0f32];
    let a_bytes = f32_slice_to_le_bytes(&a);
    let b_bytes = f32_slice_to_le_bytes(&b);
    let c_bytes = f32_slice_to_le_bytes(&c);

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert(
        "alpha".to_string(),
        TensorView::new(StDtype::F32, vec![2], &a_bytes).unwrap(),
    );
    tensors.insert(
        "beta".to_string(),
        TensorView::new(StDtype::F32, vec![3], &b_bytes).unwrap(),
    );
    tensors.insert(
        "gamma".to_string(),
        TensorView::new(StDtype::F32, vec![1], &c_bytes).unwrap(),
    );

    let serialized = safetensors::serialize(&tensors, &None).unwrap();
    let reader = SafetensorsReader::from_bytes(serialized).unwrap();

    assert_eq!(reader.len(), 3);
    assert!(reader.get("alpha").is_some());
    assert!(reader.get("beta").is_some());
    assert!(reader.get("gamma").is_some());
    assert!(reader.get("nonexistent").is_none());
    assert!(reader.get("").is_none());
}

#[test]
fn iter_yields_every_tensor() {
    let t1 = vec![1.0f32];
    let t2 = vec![2.0f32, 3.0];
    let t1b = f32_slice_to_le_bytes(&t1);
    let t2b = f32_slice_to_le_bytes(&t2);

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert(
        "first".to_string(),
        TensorView::new(StDtype::F32, vec![1], &t1b).unwrap(),
    );
    tensors.insert(
        "second".to_string(),
        TensorView::new(StDtype::F32, vec![2], &t2b).unwrap(),
    );

    let serialized = safetensors::serialize(&tensors, &None).unwrap();
    let reader = SafetensorsReader::from_bytes(serialized).unwrap();

    let seen: Vec<String> = reader.iter().map(|e| e.name.to_string()).collect();
    assert_eq!(seen.len(), 2);
    // HashMap insertion order is not deterministic in the safetensors
    // writer's serialize step, so just assert the set, not the order.
    assert!(seen.contains(&"first".to_string()));
    assert!(seen.contains(&"second".to_string()));
}

#[test]
fn metadata_roundtrips_when_present() {
    let data: Vec<f32> = vec![1.0, 2.0];
    let bytes = f32_slice_to_le_bytes(&data);
    let view = TensorView::new(StDtype::F32, vec![2], &bytes).unwrap();

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert("w".to_string(), view);

    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("author".to_string(), "atenia-test".to_string());
    meta.insert("version".to_string(), "1.0".to_string());

    let serialized = safetensors::serialize(&tensors, &Some(meta.clone())).unwrap();
    let reader = SafetensorsReader::from_bytes(serialized).unwrap();

    let got = reader.metadata().expect("metadata must be present");
    assert_eq!(got.len(), 2);
    assert_eq!(got.get("author"), Some(&"atenia-test".to_string()));
    assert_eq!(got.get("version"), Some(&"1.0".to_string()));
}

#[test]
fn metadata_is_none_when_absent() {
    let data: Vec<f32> = vec![1.0];
    let bytes = f32_slice_to_le_bytes(&data);
    let view = TensorView::new(StDtype::F32, vec![1], &bytes).unwrap();

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert("w".to_string(), view);

    let serialized = safetensors::serialize(&tensors, &None).unwrap();
    let reader = SafetensorsReader::from_bytes(serialized).unwrap();

    assert!(reader.metadata().is_none());
}
