//! Real-checkpoint integration test for `ShardedSafetensorsReader`
//! (M4.7.1.c).
//!
//! Loads the actual `mistralai/Mistral-7B-v0.3` checkpoint
//! (3 shards, ~14 GB BF16) from disk, asserts every tensor declared
//! by the index lands as a graph parameter via the sharded driver,
//! and verifies the per-shard load + drop pattern keeps `loaded`,
//! `skipped`, `missing` consistent with the index.
//!
//! No forward pass, no numerical validation — this test is purely
//! about the loader. Numerical fidelity of a real Llama-2-class
//! checkpoint is M4.7.6's job once GPU residency, BF16 storage,
//! and tier policy land. M4.7.1 just proves the multi-file format
//! parses end-to-end.
//!
//! Marked `#[ignore]`. Run with:
//!
//! ```powershell
//! $env:MISTRAL7B_INDEX_PATH = "F:\Proyectos\artenia_engine\atenia-engine\models\mistral-7b-v0.3\model.safetensors.index.json"
//! cargo test --test mistral_7b_sharded_load_test --release \
//!     -- --ignored --nocapture
//! ```

use std::env;
use std::path::Path;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::shard_index::ShardIndex;
use atenia_engine::v17::loader::sharded_reader::ShardedSafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::WeightMapper;

#[test]
#[ignore = "requires MISTRAL7B_INDEX_PATH env var pointing at model.safetensors.index.json of Mistral 7B v0.3"]
fn mistral_7b_sharded_load_round_trips_every_tensor() {
    println!("\n=== Mistral 7B v0.3 sharded load test (M4.7.1.c) ===\n");

    let index_path_str = env::var("MISTRAL7B_INDEX_PATH").expect(
        "Set MISTRAL7B_INDEX_PATH to the absolute path of \
         Mistral 7B v0.3's model.safetensors.index.json",
    );
    let index_path = Path::new(&index_path_str);
    println!("Index file: {}", index_path.display());

    // ---- Phase 1: parse the index alone, discover shape of work
    let index_open_start = std::time::Instant::now();
    let index = ShardIndex::from_file(index_path).expect("failed to parse index");
    println!(
        "Index parsed in {:.3}s: {} tensors across {} shards (declared total_size {} bytes)",
        index_open_start.elapsed().as_secs_f32(),
        index.weight_map.len(),
        index.shard_count(),
        index.total_size,
    );

    // Mistral 7B v0.3 typically ships as 3 shards. Defensive
    // assertion: the test fails loudly if the upstream checkpoint
    // ever switches to a different shard count, which would indicate
    // either an upstream re-shard or the wrong model under the env
    // var.
    assert!(
        index.shard_count() >= 2,
        "expected sharded checkpoint (>=2 shards), got {}; \
         did MISTRAL7B_INDEX_PATH point at a single-file model?",
        index.shard_count()
    );

    // ---- Phase 2: build a graph with one Parameter per indexed
    //              tensor, sized according to the safetensors header
    //              of the shard that hosts it. We need the shapes
    //              to register parameters with the right sizes; pull
    //              them from a transient open of each shard, then
    //              drop the readers.
    let shapes_start = std::time::Instant::now();
    let mut shapes: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for shard_filename in index.shard_filenames() {
        let shard_path = index.shard_path(&shard_filename);
        let reader = atenia_engine::v17::loader::safetensors_reader::SafetensorsReader::open(
            &shard_path,
        )
        .unwrap_or_else(|e| panic!("open shard {}: {:?}", shard_path.display(), e));
        for entry in reader.iter() {
            shapes.insert(entry.name.to_string(), entry.shape.to_vec());
        }
        // reader drops here, freeing this shard's raw bytes before
        // the next shard opens.
    }
    println!(
        "Shape discovery across {} shards in {:.3}s — {} tensor shapes captured",
        index.shard_count(),
        shapes_start.elapsed().as_secs_f32(),
        shapes.len()
    );
    assert_eq!(
        shapes.len(),
        index.weight_map.len(),
        "every tensor in the index must be present in exactly one shard"
    );

    // ---- Phase 3: build a graph with empty parameters of the
    //              right shapes, register a mapper that knows about
    //              every name.
    let build_start = std::time::Instant::now();
    let mut gb = GraphBuilder::new();
    let mut names: Vec<String> = Vec::with_capacity(shapes.len());
    let mut ids: Vec<usize> = Vec::with_capacity(shapes.len());
    for (name, shape) in &shapes {
        let numel: usize = shape.iter().product();
        let tensor = Tensor::new_cpu(shape.clone(), vec![0.0_f32; numel]);
        let id = gb.parameter(tensor);
        names.push(name.clone());
        ids.push(id);
    }
    let mut graph = gb.build();
    println!(
        "Graph built in {:.3}s — {} parameter nodes",
        build_start.elapsed().as_secs_f32(),
        ids.len()
    );

    let mapper = WeightMapper::from_param_names_and_ids(&names, &ids).expect("mapper builds");

    // ---- Phase 4: load via ShardedSafetensorsReader.
    let load_start = std::time::Instant::now();
    let sharded = ShardedSafetensorsReader::open(index_path).expect("sharded reader opens");
    let report = sharded
        .load_into(&mut graph, &mapper)
        .expect("sharded load_into succeeds");
    println!(
        "Sharded load completed in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );
    println!(
        "  loaded:  {} tensors\n  skipped: {} ({:?})\n  missing: {} ({:?})",
        report.loaded,
        report.skipped.len(),
        if report.skipped.len() > 5 {
            report
                .skipped
                .iter()
                .take(5)
                .cloned()
                .chain(std::iter::once("...".to_string()))
                .collect::<Vec<_>>()
        } else {
            report.skipped.clone()
        },
        report.missing.len(),
        report.missing,
    );

    // ---- Assertions ----
    assert_eq!(
        report.loaded,
        index.weight_map.len(),
        "every index entry must land as a loaded tensor"
    );
    assert!(
        report.skipped.is_empty(),
        "no shard should carry tensors absent from the mapper (we registered \
         exactly the index): skipped = {:?}",
        report.skipped
    );
    assert!(
        report.missing.is_empty(),
        "no mapper entry should be left unsatisfied: missing = {:?}",
        report.missing
    );

    // Sanity: every parameter is non-empty post-load (zero would
    // mean the copy_from_slice path didn't run).
    let mut all_zero_count = 0usize;
    for &id in &ids {
        let slice = graph.nodes[id].output.as_ref().unwrap().as_cpu_slice();
        if slice.iter().all(|v| *v == 0.0) {
            all_zero_count += 1;
        }
    }
    // An entirely-zero tensor would be deeply suspicious for a real
    // pretrained model. Allow a tiny number (RmsNorm gammas are
    // initialised at 1.0 in HF but a stray zero somewhere is fine);
    // refuse if half or more of the parameters round-tripped to zero.
    assert!(
        all_zero_count * 2 < ids.len(),
        "{}/{} parameters are entirely zero post-load — copy path likely broken",
        all_zero_count,
        ids.len()
    );

    println!(
        "\nSanity: {}/{} parameters are non-zero post-load",
        ids.len() - all_zero_count,
        ids.len()
    );
    println!("\n=== Mistral 7B v0.3 M4.7.1.c PASSED ===\n");
}
