//! **FP8-SAFETENSORS-1** — FP8 safetensors read validation.
//!
//! Reads real FP8 safetensors fixtures (`F8_E4M3` / `F8_E5M2`, produced by
//! PyTorch `torch.float8_e4m3fn` / `e5m2` + `safetensors.torch.save_file`) and
//! asserts the reader's FP8→F32 decode is **bit-identical** to PyTorch's own
//! `fp8.to(float32)` upcast (committed as a sibling F32 reference). Also
//! asserts the reader presents the FP8 tensor as a plain **F32** entry — so the
//! graph / kernels / adapters never see an FP8 dtype.

use std::path::PathBuf;

use atenia_engine::tensor::DType;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fp8")
        .join(name)
}

fn read_f32(file: &str, tensor: &str) -> (Vec<f32>, DType, Vec<usize>) {
    let r = SafetensorsReader::open(&fixture(file)).expect("open fixture");
    let e = r.get(tensor).expect("tensor present");
    (e.to_vec_f32().expect("decode f32"), e.dtype, e.shape.to_vec())
}

#[test]
fn e4m3_decodes_bit_exact_vs_torch() {
    let (got, dtype, shape) = read_f32("tiny_e4m3.safetensors", "w_e4m3");
    let (reference, _, ref_shape) = read_f32("tiny_e4m3_ref_f32.safetensors", "w_e4m3");
    // The reader must present the FP8 tensor as F32 (decoded at construction).
    assert_eq!(dtype, DType::F32, "FP8 tensor must surface as F32");
    assert_eq!(shape, ref_shape);
    assert_eq!(got, reference, "E4M3 decode must match torch fp8->f32 bit-exactly");
}

#[test]
fn e5m2_decodes_bit_exact_vs_torch() {
    let (got, dtype, shape) = read_f32("tiny_e5m2.safetensors", "w_e5m2");
    let (reference, _, ref_shape) = read_f32("tiny_e5m2_ref_f32.safetensors", "w_e5m2");
    assert_eq!(dtype, DType::F32, "FP8 tensor must surface as F32");
    assert_eq!(shape, ref_shape);
    assert_eq!(got, reference, "E5M2 decode must match torch fp8->f32 bit-exactly");
}

#[test]
fn fp8_iter_reports_f32_not_fp8() {
    let r = SafetensorsReader::open(&fixture("tiny_e4m3.safetensors")).expect("open");
    for e in r.iter() {
        assert_eq!(e.dtype, DType::F32, "iter must report decoded F32 for FP8 tensors");
        // 12 elements × 4 bytes (decoded), not 12 × 1 (on-disk FP8).
        assert_eq!(e.raw_bytes.len(), 12 * 4);
    }
}

// End-to-end: a full model whose weights are stored as FP8 (E4M3) safetensors
// loads through the pipeline and generates. FP8 is lossy, so we assert it
// *loads and runs* (non-empty output, no error), not text parity with the
// f16 original. `#[ignore]` — needs a local FP8 model dir (convert a small
// model: `state_dict[k].to(torch.float8_e4m3fn)` → save_file).
//   $env:SMOLLM_FP8_DIR = "<tmp>/smollm2_135m_fp8"
#[test]
#[ignore = "needs SMOLLM_FP8_DIR (a model.safetensors with F8_E4M3 weights)"]
fn fp8_model_loads_and_generates() {
    use atenia_engine::nn::llama::generator::{GeneratedToken, TokenSink};
    use atenia_engine::nn::llama::pipeline::GenerationPipeline;

    struct Null;
    impl TokenSink for Null {
        fn on_token(&mut self, _t: &GeneratedToken) {}
    }

    let dir = std::env::var("SMOLLM_FP8_DIR").expect("set SMOLLM_FP8_DIR");
    let p = GenerationPipeline::from_model_dir(&dir).expect("load FP8 model");
    let out = p
        .generate_raw("The capital of France is", 12, &mut Null)
        .expect("generate from FP8 weights");
    println!("FP8 model output: {out:?}");
    assert!(!out.trim().is_empty(), "FP8 model must produce some output");
}
