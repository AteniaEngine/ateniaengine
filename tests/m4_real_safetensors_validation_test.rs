//! Empirical validation of the M4 loader against a real HuggingFace
//! safetensors file (gpt2 base model, ~548 MB F32).
//!
//! Every other M4 test uses synthetic data — bytes produced by
//! Atenia's own call to `safetensors::serialize` on in-memory tensors.
//! That validates the reader against the writer in the same crate
//! version, but does not exercise a file as a third-party producer
//! would emit one.
//!
//! This test opens a real checkpoint and walks the reader API end to
//! end: parse header, enumerate tensors, inspect shapes and dtypes,
//! extract one well-known tensor (the token embedding) and validate
//! its values are finite and within a reasonable numeric range.
//!
//! Marked `#[ignore]` because the file is ~548 MB and not shipped
//! with the repo. Run manually with:
//!
//!     # PowerShell
//!     $env:GPT2_SAFETENSORS_PATH = "F:\path\to\model.safetensors"
//!     cargo test --test m4_real_safetensors_validation_test -- --ignored --nocapture
//!
//!     # Bash
//!     GPT2_SAFETENSORS_PATH=/path/to/model.safetensors \
//!         cargo test --test m4_real_safetensors_validation_test -- --ignored --nocapture
//!
//! The `--nocapture` flag surfaces the `println!` lines so you can
//! see the tensor count, dtype distribution, embedding shape, and
//! numeric statistics. Without `--nocapture` the test still runs
//! but the diagnostics are hidden by cargo's default capture.
//!
//! This test does NOT execute the model. That requires M4.5
//! (RoPE / gpt2-specific graph builder / numerical validation
//! against a reference).

use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use std::env;
use std::path::Path;

#[test]
#[ignore = "requires GPT2_SAFETENSORS_PATH env var pointing to gpt2 model.safetensors"]
fn validate_gpt2_safetensors_real_world() {
    let path = env::var("GPT2_SAFETENSORS_PATH")
        .expect("Set GPT2_SAFETENSORS_PATH to the gpt2 model.safetensors path");

    println!("\n=== GPT-2 Safetensors Validation ===");
    println!("Path: {}", path);

    // 1. Parse header without crashing.
    let reader =
        SafetensorsReader::open(Path::new(&path)).expect("failed to parse gpt2 safetensors");
    println!("+ Parsed header successfully");

    // 2. Enumeration returns a non-trivial number of tensors.
    let entries: Vec<_> = reader.iter().collect();
    assert!(
        entries.len() > 100,
        "expected >100 tensors in gpt2, got {}",
        entries.len()
    );
    println!("+ Found {} tensors", entries.len());

    // 3. Shapes are structurally valid: non-empty, all dims > 0.
    let mut total_elements: u64 = 0;
    let mut dtype_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

    for entry in &entries {
        assert!(
            !entry.shape.is_empty(),
            "tensor '{}' has empty shape",
            entry.name
        );
        assert!(
            entry.shape.iter().all(|&d| d > 0),
            "tensor '{}' has zero dimension (shape={:?})",
            entry.name,
            entry.shape
        );

        let numel: u64 = entry.shape.iter().map(|&d| d as u64).product();
        total_elements += numel;
        *dtype_counts
            .entry(format!("{:?}", entry.dtype))
            .or_insert(0u32) += 1;
    }
    println!("+ All shapes valid (non-empty, all dims > 0)");
    println!("  Total elements across all tensors: {}", total_elements);
    println!("  Dtype distribution: {:?}", dtype_counts);

    // 4. Locate the token embedding. Naming varies by producer:
    // HF-Transformers canonical is "wte.weight"; some exports prefix
    // with "transformer.". Try the common variants.
    let embedding_names = ["wte.weight", "transformer.wte.weight", "wte"];
    let mut found_embedding = None;
    for name in &embedding_names {
        if let Some(entry) = reader.get(name) {
            found_embedding = Some((name.to_string(), entry));
            break;
        }
    }

    let (emb_name, wte) = found_embedding.expect(
        "failed to find token embedding in gpt2 (tried wte.weight, \
         transformer.wte.weight, wte)",
    );
    println!("+ Found embedding at '{}'", emb_name);
    println!("  Shape: {:?}", wte.shape);

    // 5. Conversion to Vec<f32> returns the expected element count.
    let values = wte
        .to_vec_f32()
        .expect("failed to convert embedding to f32");
    let expected_numel: usize = wte.shape.iter().product();
    assert_eq!(
        values.len(),
        expected_numel,
        "length mismatch: got {}, expected {}",
        values.len(),
        expected_numel
    );
    println!("+ Converted to Vec<f32>: {} elements", values.len());

    // 6. Finiteness: no NaN or inf in the embedding.
    let finite_count = values.iter().filter(|v| v.is_finite()).count();
    assert_eq!(
        finite_count,
        values.len(),
        "{} non-finite values in embedding",
        values.len() - finite_count
    );
    println!("+ All values finite");

    // 7. Numeric range sanity: trained embeddings should stay in a
    // small range. A max absolute value above 10 would indicate
    // decode corruption or a wildly mis-trained model.
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let min_abs_nonzero = values
        .iter()
        .map(|v| v.abs())
        .filter(|&v| v > 0.0)
        .fold(f32::INFINITY, f32::min);
    let mean_abs: f32 = values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32;

    assert!(
        max_abs < 10.0,
        "embedding max absolute value suspiciously large: {}",
        max_abs
    );
    println!("+ Value range reasonable:");
    println!("  max |v|:         {:.6}", max_abs);
    println!("  min nonzero |v|: {:.6}", min_abs_nonzero);
    println!("  mean |v|:        {:.6}", mean_abs);

    // 8. Global metadata (optional; not all producers include it).
    match reader.metadata() {
        Some(meta) => {
            println!("+ File has metadata with {} keys:", meta.len());
            for (k, v) in meta.iter().take(5) {
                println!("  {}: {}", k, v);
            }
        }
        None => {
            println!("+ File has no __metadata__ section (normal for some producers)");
        }
    }

    println!("\n=== All validations passed ===\n");
}
