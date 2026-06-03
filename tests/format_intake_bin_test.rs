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

use atenia_engine::v17::loader::pytorch_bin::transcode_bin_to_safetensors;
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
