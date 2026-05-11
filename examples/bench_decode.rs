//! M5.f.a — decode-step micro-bench harness (D68).
//!
//! Component-wise wall-clock breakdown of one greedy decode
//! step on the M5 cache-aware attention path. Establishes
//! the empirical baseline the M6 GPU acceleration work will
//! be measured against.
//!
//! ## What this measures
//!
//! Per layer, at the M=1 shapes the decode loop actually
//! runs against:
//!
//!   - Q, K, V projection — `[1, hidden] @ [hidden, hidden]`
//!     (post-tile MHA shape; gap-3 Way A confirmed by M5.d.b).
//!   - Attention scores — `[1, n_heads, 1, head_dim] @
//!     [1, n_heads, head_dim, cached_len]`.
//!   - Attention output — `[1, n_heads, 1, cached_len] @
//!     [1, n_heads, cached_len, head_dim]`.
//!   - FFN gate, up, down — three matmuls at intermediate
//!     dim with SwiGLU between gate and up.
//!
//! ## What this does NOT measure
//!
//! The **per-step graph rebuild cost** — currently the
//! M5.c.2.c-locked policy rebuilds the whole decode graph at
//! every step. On 13B that build is ~2.1s (M4.7.6.d), which
//! means every decode token pays ~2.1s of pure construction
//! overhead before any compute lands. The 0.06 tok/s
//! observed in M5.e on Llama 2 13B Chat is dominated by this
//! rebuild, not by the math measured here.
//!
//! M6 will collapse the rebuild cost to zero by reusing one
//! decode graph across steps (fixed-size cache + valid_len
//! mask, populated via `Graph::overwrite_parameter`). This
//! bench gives M6 the per-component target: how fast can
//! the decode math go when the rebuild penalty disappears.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --example bench_decode -- [model_dir]
//! ```
//!
//! Defaults to `models/tinyllama-1.1b`. Override by passing
//! a model dir as the only positional arg, or by setting
//! `ATENIA_BENCH_MODEL_DIR`. The model is loaded once via
//! `GenerationPipeline::from_model_dir` (the same path the
//! `atenia generate` CLI uses) so the timings reflect what
//! the user actually pays in production.

use std::time::Instant;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::kv_cache::KvCacheBuildSpec;
use atenia_engine::nn::llama::{GenerationPipeline, LlamaRuntime, build_llama_with_store};
use atenia_engine::tensor::Tensor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("ATENIA_BENCH_MODEL_DIR").ok())
        .unwrap_or_else(|| "models/tinyllama-1.1b".to_string());

    eprintln!("=== Atenia decode-step micro-bench (M5.f.a / D68) ===");
    eprintln!("model dir: {model_dir}");
    eprintln!();

    let load_start = Instant::now();
    eprintln!("loading model ...");
    let pipe = match GenerationPipeline::from_model_dir(&model_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: failed to load model: {e}");
            std::process::exit(1);
        }
    };
    let load_secs = load_start.elapsed().as_secs_f32();
    let resident_gib = pipe.store.resident_bytes() as f64 / (1024.0_f64.powi(3));
    eprintln!(
        "loaded in {:.1}s ({} parameters, {:.2} GiB resident)",
        load_secs,
        pipe.store.len(),
        resident_gib
    );
    eprintln!();
    eprintln!(
        "config: {} layers, {} attention heads, {} kv heads, hidden {}, head_dim {}, intermediate {}",
        pipe.config.num_hidden_layers,
        pipe.config.num_attention_heads,
        pipe.config.num_key_value_heads,
        pipe.config.hidden_size,
        pipe.config.effective_head_dim(),
        pipe.config.intermediate_size,
    );
    eprintln!();

    // ---- End-to-end decode-step bench ----
    //
    // We run a *real* decode-step build + forward at
    // cached_len = 8 (a representative mid-prompt cache
    // length) and measure each phase independently.
    let cached_len = 8usize;
    eprintln!("=== Per-step bench (cached_len = {cached_len}) ===");

    // 1. Graph build cost (the M6 target — dominates 13B
    //    today).
    let build_start = Instant::now();
    let mut gb = GraphBuilder::new();
    let token_in = gb.input();
    let runtime = LlamaRuntime { batch: 1, seq: 1 };
    let spec = KvCacheBuildSpec { cached_len };
    let h = build_llama_with_store(
        &mut gb,
        &pipe.config,
        &runtime,
        token_in,
        &pipe.store,
        Some(&spec),
    )
    .expect("decode build must succeed");
    let _ = gb.output(h.logits_id);
    let mut g = gb.build();
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("graph build:           {build_ms:>10.2} ms");

    // 2. Cache-slot patch cost (overwrite_parameter per
    //    layer × 2). Should be near zero — it's a single
    //    tensor swap per slot, no math.
    let kv_handles = h.kv_handles.as_ref().expect("kv_handles must be Some");
    let cache_shape = vec![
        1,
        pipe.config.num_attention_heads,
        cached_len,
        pipe.config.effective_head_dim(),
    ];
    let cache_numel: usize = cache_shape.iter().product();
    let zero_cache = Tensor::new_cpu(cache_shape.clone(), vec![0.0_f32; cache_numel]);
    let patch_start = Instant::now();
    for layer in &kv_handles.per_layer {
        g.overwrite_parameter(layer.cache_k_param_id, zero_cache.clone())
            .unwrap();
        g.overwrite_parameter(layer.cache_v_param_id, zero_cache.clone())
            .unwrap();
    }
    let patch_ms = patch_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "cache slot patches:    {patch_ms:>10.2} ms ({} slots = {} layers × 2)",
        kv_handles.per_layer.len() * 2,
        kv_handles.per_layer.len()
    );

    // 3. Forward execute cost (pure math: Q/K/V/O projections,
    //    Concat with cache, attention BMM, softmax, V·attn,
    //    SwiGLU, residual + lm_head).
    let token_input = Tensor::new_cpu(vec![1, 1], vec![0.0_f32]);
    let fwd_start = Instant::now();
    let logits = g
        .execute(vec![token_input])
        .into_iter()
        .next()
        .expect("forward output");
    let fwd_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("forward execute:       {fwd_ms:>10.2} ms");
    eprintln!();

    let total_ms = build_ms + patch_ms + fwd_ms;
    eprintln!(
        "--- step total:        {total_ms:>10.2} ms ({:.2} tok/s)",
        1000.0 / total_ms
    );
    eprintln!();

    // ---- Component-wise estimates (informational) ----
    //
    // We don't have per-op tracing infrastructure in the
    // executor today, so this section reports the shapes
    // and theoretical FLOP counts so M6 has a target to
    // beat. The actual measured forward cost is the
    // `fwd_ms` figure above.
    eprintln!("=== Theoretical FLOP estimates per layer ===");
    let h_size = pipe.config.hidden_size;
    let i_size = pipe.config.intermediate_size;
    let n_layers = pipe.config.num_hidden_layers;

    let qkvo_flops = 4 * 2 * h_size * h_size; // 4 matmuls, 2 FLOPs/elt
    let ffn_flops = 3 * 2 * h_size * i_size;
    let attn_flops = 2 * 2 * h_size * (cached_len + 1); // scores + output
    let layer_flops = qkvo_flops + ffn_flops + attn_flops;
    let total_flops = layer_flops * n_layers;

    eprintln!("per layer:");
    eprintln!(
        "  Q+K+V+O matmuls:     {:>12} FLOPs ({} × 2·hidden² at M=1)",
        qkvo_flops, 4
    );
    eprintln!(
        "  FFN gate+up+down:    {:>12} FLOPs ({} × 2·hidden·intermediate at M=1)",
        ffn_flops, 3
    );
    eprintln!(
        "  attention scores+out:{:>12} FLOPs (2× M=1 × hidden × cached_len)",
        attn_flops
    );
    eprintln!("total layer:           {:>12} FLOPs", layer_flops);
    eprintln!(
        "total step ({} layers): {:>12} FLOPs",
        n_layers, total_flops
    );
    eprintln!();
    let gflops = total_flops as f64 / (fwd_ms / 1000.0) / 1e9;
    eprintln!("=> measured throughput: {gflops:.2} GFLOPS over the forward execute");
    eprintln!();

    // ---- Headroom analysis ----
    eprintln!("=== Bottleneck analysis ===");
    let build_pct = build_ms / total_ms * 100.0;
    let fwd_pct = fwd_ms / total_ms * 100.0;
    eprintln!("build:    {build_pct:>5.1}%  ({build_ms:.0} ms / {total_ms:.0} ms total)");
    eprintln!("forward:  {fwd_pct:>5.1}%  ({fwd_ms:.0} ms / {total_ms:.0} ms total)");
    eprintln!();
    if build_pct > 50.0 {
        eprintln!("M6 priority: graph rebuild dominates ({build_pct:.0}%).");
        eprintln!("  Single decode-graph reuse + valid_len mask should eliminate");
        eprintln!("  this overhead and shift the bottleneck to forward compute.");
    } else {
        eprintln!("M6 priority: forward compute dominates ({fwd_pct:.0}%).");
        eprintln!("  GPU offload (cuda_matmul non-pooled + per-layer streaming)");
        eprintln!("  is the next lever.");
    }

    // Suppress unused-warning on logits.
    let _ = logits;
}
