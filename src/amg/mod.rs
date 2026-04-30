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

// M5.b — KV cache runtime infrastructure.
// Owns the third tensor category (D59 — Parameter /
// Activation / KvCache), the runtime KvCache data structure
// (D60 mutable cells, append/get/clear), and the placeholder
// handle types M5.c will populate when it extends
// `build_llama` to thread cache awareness through the
// attention path.
pub mod kv_cache;
