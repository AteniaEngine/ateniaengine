//! **FORMAT-INTAKE-1** — PyTorch `.bin` transcoder round-trip.
//!
//! Reads a real `torch.save` checkpoint (`tests/fixtures/pytorch_bin/tiny.bin`,
//! committed) and asserts the transcoded safetensors is **byte-identical** —
//! name set, per-tensor shape, dtype, and raw bytes — to a reference
//! safetensors saved from the same state dict
//! (`tiny_reference.safetensors`). This validates the hand-rolled ZIP reader +
//! restricted unpickler end to end, in CI, with no torch dependency at test
//! time. Covers F32 (2D), F16 (2D), BF16 (1D), and a second F32 tensor.

use std::collections::BTreeMap;
use std::path::PathBuf;

use atenia_engine::v17::loader::pytorch_bin::{
    transcode_bin_to_safetensors, transcode_sharded_bin_to_safetensors,
};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pytorch_bin")
        .join(name)
}

fn collect(reader: &SafetensorsReader) -> BTreeMap<String, (Vec<usize>, String, Vec<u8>)> {
    reader
        .iter()
        .map(|e| {
            (
                e.name.to_string(),
                (e.shape.to_vec(), format!("{:?}", e.dtype), e.raw_bytes.to_vec()),
            )
        })
        .collect()
}

#[test]
fn bin_transcode_matches_reference_safetensors() {
    let bin_bytes = std::fs::read(fixture("tiny.bin")).expect("read tiny.bin");
    let transcoded = transcode_bin_to_safetensors(&bin_bytes).expect("transcode .bin");
    let from_bin = SafetensorsReader::from_bytes(transcoded).expect("read transcoded safetensors");

    let reference = SafetensorsReader::open(&fixture("tiny_reference.safetensors"))
        .expect("read reference safetensors");

    let a = collect(&from_bin);
    let b = collect(&reference);

    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
        "tensor name sets differ"
    );
    for (name, (shape_b, dtype_b, bytes_b)) in &b {
        let (shape_a, dtype_a, bytes_a) = a.get(name).expect("tensor present in transcoded");
        assert_eq!(shape_a, shape_b, "{name}: shape mismatch");
        assert_eq!(dtype_a, dtype_b, "{name}: dtype mismatch");
        assert_eq!(bytes_a, bytes_b, "{name}: raw bytes mismatch");
    }
    assert_eq!(a.len(), 4, "expected 4 tensors");
}

#[test]
fn corrupt_bin_fails_loud() {
    // A truncated copy of the real .bin must error, not panic or silently load.
    let mut bin = std::fs::read(fixture("tiny.bin")).expect("read tiny.bin");
    bin.truncate(bin.len() / 2);
    let r = transcode_bin_to_safetensors(&bin);
    assert!(r.is_err(), "a truncated .bin must be rejected");
}

#[test]
fn non_bin_bytes_fail_loud() {
    let r = transcode_bin_to_safetensors(b"this is plainly not a pytorch checkpoint");
    assert!(r.is_err(), "non-zip bytes must be rejected with an error");
}

// ---------------------------------------------------------------------------
// FORMAT-INTAKE-2 — sharded `.bin`
// ---------------------------------------------------------------------------

fn shard_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pytorch_bin_sharded")
        .join(name)
}

fn scratch_dir(label: &str) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let d = std::env::temp_dir().join(format!("atenia_fi2_{label}_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

const SHARD1: &str = "pytorch_model-00001-of-00002.bin";
const SHARD2: &str = "pytorch_model-00002-of-00002.bin";

#[test]
fn sharded_bin_transcode_matches_reference() {
    let index = shard_fixture("pytorch_model.bin.index.json");
    let transcoded = transcode_sharded_bin_to_safetensors(&index).expect("transcode sharded .bin");
    let from_bin = SafetensorsReader::from_bytes(transcoded).expect("read transcoded");

    let reference = SafetensorsReader::open(&shard_fixture("assembled_reference.safetensors"))
        .expect("read reference");

    let a = collect(&from_bin);
    let b = collect(&reference);
    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
        "assembled tensor set differs"
    );
    for (name, (shape_b, dtype_b, bytes_b)) in &b {
        let (shape_a, dtype_a, bytes_a) = a.get(name).expect("present");
        assert_eq!(shape_a, shape_b, "{name}: shape");
        assert_eq!(dtype_a, dtype_b, "{name}: dtype");
        assert_eq!(bytes_a, bytes_b, "{name}: bytes");
    }
    assert_eq!(a.len(), 4, "expected 4 assembled tensors");
}

#[test]
fn sharded_missing_shard_fails_loud() {
    let dir = scratch_dir("missing");
    // Copy the index + only ONE of the two shards.
    std::fs::copy(shard_fixture("pytorch_model.bin.index.json"), dir.join("pytorch_model.bin.index.json")).unwrap();
    std::fs::copy(shard_fixture(SHARD1), dir.join(SHARD1)).unwrap();
    let r = transcode_sharded_bin_to_safetensors(&dir.join("pytorch_model.bin.index.json"));
    assert!(r.is_err(), "a missing shard must be rejected");
    assert!(format!("{}", r.unwrap_err()).contains("cannot read shard"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sharded_weight_map_declares_absent_tensor_fails_loud() {
    let dir = scratch_dir("absent");
    std::fs::copy(shard_fixture(SHARD1), dir.join(SHARD1)).unwrap();
    std::fs::copy(shard_fixture(SHARD2), dir.join(SHARD2)).unwrap();
    // Index that references a tensor no shard provides.
    let bad = format!(
        r#"{{"metadata":{{"total_size":0}},"weight_map":{{
            "model.embed_tokens.weight":"{SHARD1}",
            "model.layers.0.mlp.gate_proj.weight":"{SHARD1}",
            "model.norm.weight":"{SHARD2}",
            "lm_head.weight":"{SHARD2}",
            "ghost.weight":"{SHARD2}"
        }}}}"#
    );
    let idx = dir.join("pytorch_model.bin.index.json");
    std::fs::write(&idx, bad).unwrap();
    let r = transcode_sharded_bin_to_safetensors(&idx);
    assert!(r.is_err(), "a weight_map ghost tensor must be rejected");
    assert!(format!("{}", r.unwrap_err()).contains("absent from the shards"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sharded_extra_tensor_not_in_weight_map_fails_loud() {
    let dir = scratch_dir("extra");
    std::fs::copy(shard_fixture(SHARD1), dir.join(SHARD1)).unwrap();
    std::fs::copy(shard_fixture(SHARD2), dir.join(SHARD2)).unwrap();
    // Index omits one real tensor (lm_head.weight) → it becomes "extra".
    let bad = format!(
        r#"{{"metadata":{{"total_size":0}},"weight_map":{{
            "model.embed_tokens.weight":"{SHARD1}",
            "model.layers.0.mlp.gate_proj.weight":"{SHARD1}",
            "model.norm.weight":"{SHARD2}"
        }}}}"#
    );
    let idx = dir.join("pytorch_model.bin.index.json");
    std::fs::write(&idx, bad).unwrap();
    let r = transcode_sharded_bin_to_safetensors(&idx);
    assert!(r.is_err(), "an undeclared shard tensor must be rejected");
    assert!(format!("{}", r.unwrap_err()).contains("not declared in weight_map"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sharded_duplicate_tensor_across_shards_fails_loud() {
    let dir = scratch_dir("dup");
    // Use shard1 content for BOTH shard files → embed/gate appear twice.
    std::fs::copy(shard_fixture(SHARD1), dir.join(SHARD1)).unwrap();
    std::fs::copy(shard_fixture(SHARD1), dir.join(SHARD2)).unwrap();
    let bad = format!(
        r#"{{"metadata":{{"total_size":0}},"weight_map":{{
            "model.embed_tokens.weight":"{SHARD1}",
            "model.layers.0.mlp.gate_proj.weight":"{SHARD2}"
        }}}}"#
    );
    let idx = dir.join("pytorch_model.bin.index.json");
    std::fs::write(&idx, bad).unwrap();
    let r = transcode_sharded_bin_to_safetensors(&idx);
    assert!(r.is_err(), "a tensor in two shards must be rejected");
    assert!(format!("{}", r.unwrap_err()).contains("more than one shard"));
    let _ = std::fs::remove_dir_all(&dir);
}

// End-to-end: a real model loaded from `pytorch_model.bin` must generate the
// SAME greedy text as the same model loaded from safetensors. `#[ignore]`
// because it needs two local model directories (the safetensors original and a
// `.bin`-converted copy). Build the `.bin` dir by `torch.save`-ing the
// safetensors state dict (see HANDOFF_FORMAT_INTAKE_1.md).
//   $env:SMOLLM_ST_DIR  = "models/SmolLM2-135M-Instruct"
//   $env:SMOLLM_BIN_DIR = "<tmp>/smollm2_135m_bin"
#[test]
#[ignore = "needs SMOLLM_ST_DIR (safetensors) + SMOLLM_BIN_DIR (pytorch_model.bin copy)"]
fn bin_pipeline_generates_same_text_as_safetensors() {
    use atenia_engine::nn::llama::generator::{GeneratedToken, TokenSink};
    use atenia_engine::nn::llama::pipeline::GenerationPipeline;

    struct Null;
    impl TokenSink for Null {
        fn on_token(&mut self, _t: &GeneratedToken) {}
    }

    let st_dir = std::env::var("SMOLLM_ST_DIR").expect("set SMOLLM_ST_DIR");
    let bin_dir = std::env::var("SMOLLM_BIN_DIR").expect("set SMOLLM_BIN_DIR");
    let prompt = "The capital of France is";

    let st = GenerationPipeline::from_model_dir(&st_dir).expect("load safetensors model");
    let out_st = st.generate_raw(prompt, 16, &mut Null).expect("generate (st)");
    drop(st);

    let bin = GenerationPipeline::from_model_dir(&bin_dir).expect("load .bin model");
    let out_bin = bin.generate_raw(prompt, 16, &mut Null).expect("generate (bin)");

    println!("safetensors: {out_st:?}");
    println!(".bin:        {out_bin:?}");
    assert_eq!(
        out_bin, out_st,
        ".bin and safetensors must produce identical greedy text"
    );
}
