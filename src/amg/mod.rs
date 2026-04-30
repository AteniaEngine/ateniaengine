//! Atenia Model Graph definitions and execution helpers.

pub mod nodes;
pub mod builder;
pub mod graph;
pub mod scheduler;
pub mod chunking;
pub mod grad_store;
pub mod fusions;
pub mod ops;
pub mod reactive;

// M6.a — per-NodeType wall-clock accumulators for
// `Graph::execute_single`. Always-pub but the recording
// path is `cfg(feature = "bench-trace")`-gated; default
// builds pay zero overhead.
pub mod bench_trace;

// M5.c.2.a — Arc-backed shared parameter store. Lets the
// prefill and decode `Graph` instances reference the same
// physical weight bytes without doubling RAM. Tensor surface
// consumed via `TensorStorage::CpuShared` /
// `TensorStorage::CpuBf16Shared` (added in the same sub-phase).
pub mod weight_store;

// M5.b — KV cache runtime infrastructure.
// Owns the third tensor category (D59 — Parameter /
// Activation / KvCache), the runtime KvCache data structure
// (D60 mutable cells, append/get/clear), and the placeholder
// handle types M5.c will populate when it extends
// `build_llama` to thread cache awareness through the
// attention path.
pub mod kv_cache;
