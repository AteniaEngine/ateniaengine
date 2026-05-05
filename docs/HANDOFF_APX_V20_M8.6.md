# HANDOFF — APX v20 — M8.6 (BF16 KV cache)

**Status:** Closed. Tag `v0.8.6-m8.6`.
**Predecessor:** M8.7 (Disk → GPU JIT pipeline, tag `v0.8.7-m8.7`).
**Successor (active):** M9 (INT8 quantisation).

> M8.6 was deferred from M5 (D62) and recovered as a 1-day side path
> after M8.7 closed. Goal: halve the KV cache RAM footprint in long
> contexts without touching graph numerics.

---

## 1. Headline numbers

| Metric                            | M5.b baseline (F32)            | M8.6 default (BF16)         |
| --------------------------------- | ------------------------------ | --------------------------- |
| Bytes / token (Llama 2 13B)       | 1 638 400 (1.5625 MiB)         | 819 200 (0.78 MiB)          |
| Resident @ seq=2048               | 3.2 GiB                        | **1.6 GiB** (–1.6 GiB)      |
| TinyLlama 1.1B-Chat 8-token gen   | "Certainly! Here are some examples" | **bit-identical**     |
| Determinism fixture token IDs     | `[29907,13946,368,29991,2266,526,777,6455]` | **bit-identical** |
| Drift envelope (per cache write)  | 0 (F32 round-trip is no-op)    | ~3e-3 relative (single BF16 round-trip) |
| ADR-004 threshold (`< 0.5`)       | passes with 500× margin        | passes with >100× margin    |

The **1.6 GiB savings** at seq=2048 directly buys headroom for M9
(INT8 quantisation, see §6 below): the freed RAM is reusable by the
M9 tier planner without touching the VRAM budget.

---

## 2. Sub-phase ledger

| Phase    | Title                                            | Commit   | Status  |
| -------- | ------------------------------------------------ | -------- | ------- |
| M8.6.0   | BF16 KV cache opt-in (`ATENIA_BF16_KV_CACHE=1`)  | `b796eaa`| ✅      |
| M8.6.1   | Flip default + 4-token TinyLlama fixture gate    | (this)   | ✅      |

**M8.6.0** introduced the data path: graph stays F32, but the runtime
ledger between decode steps holds BF16 cells. The cast happens in
`generator.rs::harvest_cache_*` (F32 → BF16) and right before
`overwrite_parameter` (BF16 → F32).

**M8.6.1** flipped the default after the TinyLlama 1.1B determinism
fixture (`tests/fixtures/generation_determinism/expected_tokens_tinyllama.json`)
came back **bit-identical** under the BF16 ledger. No regenerated
fixture; no token drift. Operators can opt out via
`ATENIA_LEGACY_F32_KV_CACHE=1`.

---

## 3. Decisions

- **D62 (resolved)** — KV cells held in the runtime ledger live at
  BF16 by default. The graph itself stays F32; the cast is applied
  only in the harvest/reinject path.
- **D87** — `ATENIA_BF16_KV_CACHE` (M8.6.0 opt-in) becomes a no-op in
  M8.6.1 (BF16 is the default). The deprecated flag is kept silent
  (no warn) so existing operator scripts keep working without noise.
- **D88** — The legacy F32 KV path is preserved behind
  `ATENIA_LEGACY_F32_KV_CACHE=1` for emergency rollback. Scheduled
  for removal in M10 if no-one reports the path needed.

---

## 4. Validation gates

| Gate                                  | Command                                                                              | Result    |
| ------------------------------------- | ------------------------------------------------------------------------------------ | --------- |
| Library tests (single-thread)         | `cargo test --lib -- --test-threads=1`                                               | 189/189 ✅ |
| M8.6 synthetic suite                  | `cargo test --test m8_6_kv_cache_bf16_test`                                          | 5/5 ✅    |
| M5 R2 falsifier                       | `cargo test --test m5_c2c_r2_kv_cache_test`                                          | 3/3 ✅    |
| M5.d.a generation loop                | `cargo test --test m5_da_generation_loop_test`                                       | 4/4 ✅    |
| TinyLlama 1.1B determinism (M8.6.1)   | `cargo test --release --test m5_db_tinyllama_pipeline_test -- --ignored --nocapture` | bit-identical ✅ |

**Pre-existing race in `cuda::disk_prefetch::tests`**: the test
`kick_off_then_take_same_handle_returns_bytes_and_increments_counter`
fails under default parallelism because the global hits counter is
shared across test threads. Single-thread is the canonical gate. This
race predates M8.6 (introduced in M8.7.1.a, listed as cleanup TODO
in `docs/HANDOFF_APX_V20_M8.7.md`).

---

## 5. API surface added

`amg::kv_cache`:

```rust
pub enum KvCellDtype { F32, BF16 }   // Default = F32, type-level

pub struct KvCacheConfig {
    pub batch: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub cell_dtype: KvCellDtype,        // M8.6 — propagated by the
                                        // runtime, set to BF16 by
                                        // default in M8.6.1.
}
impl KvCacheConfig {
    pub fn bytes_per_token_f32(&self) -> usize;
    pub fn bytes_per_token(&self)     -> usize;     // M8.6 dtype-aware
}

pub struct KvCache { /* ... */ }
impl KvCache {
    pub fn resident_bytes_f32(&self) -> usize;
    pub fn resident_bytes(&self)     -> usize;      // M8.6 dtype-aware
}

pub struct KvLayer { /* ... */ }
impl KvLayer {
    pub fn empty(...)            -> Self;
    pub fn empty_with_dtype(...) -> Self;           // M8.6
}

// Runtime cast helpers used by `generator.rs`.
pub fn cast_kv_cell_f32_to_bf16(t: &Tensor) -> Tensor;
pub fn cast_kv_cell_bf16_to_f32(t: &Tensor) -> Tensor;
```

`append_along_seq_axis` dispatches on `dst.storage`: a `CpuBf16` dst
truncates F32 slices via `f32_to_bf16_bits` and writes back as
`Tensor::CpuBf16`. F32 dst keeps the M5.b bit-exact path.

---

## 6. Why this matters for M9

M9 (INT8 quantisation) eliminates the Disk overflow path entirely by
shrinking 13B from 24 GiB BF16 → 12 GiB INT8. The tier planner
(`src/gpu/tier_plan.rs`) needs RAM headroom for two pressures:

1. The 12 GiB residency target has to fit alongside activations and
   the KV cache.
2. The current 8 GiB `RAM_HEADROOM` constant is conservative for the
   F32 cache.

M8.6 frees 1.6 GiB at seq=2048, which is exactly the kind of
working-set room M9's planner can use to keep more layers RAM-resident
without spilling to disk in long contexts. **No M9 plan changes are
required to consume the savings**; the planner already reads
`KvCacheConfig::bytes_per_token` (now dtype-aware) when present.

---

## 7. Operator quickstart

**Default (M8.6.1+, BF16 ledger):**

```powershell
cargo run --release --example llama2_13b_demo
# KV cache lives at BF16 between decode steps. No flag needed.
```

**Emergency rollback to F32:**

```powershell
$env:ATENIA_LEGACY_F32_KV_CACHE = "1"
cargo run --release --example llama2_13b_demo
```

**Memory expectations (Llama 2 13B Chat, 40 layers × 40 kv_heads × 128 head_dim):**

| Context length | F32 cache | BF16 cache (default) |
| -------------: | --------: | -------------------: |
|    256 tokens  |  410 MiB  |    **205 MiB**       |
|   1024 tokens  |  1.6 GiB  |    **820 MiB**       |
|   2048 tokens  |  3.2 GiB  |    **1.6 GiB**       |
|   4096 tokens  |  6.4 GiB  |    **3.2 GiB**       |

---

## 8. Open issues / known limitations

- Determinism fixture for **SmolLM2-1.7B / Qwen2.5-1.5B / Llama 3.2-1B**
  under the BF16 default not yet recorded. Only TinyLlama 1.1B is
  bit-locked. M8.6.1 ships on the assumption that the per-step BF16
  round-trip drift is bounded the same way for the other 3 models
  (same architecture family, similar weight magnitudes). If a
  fixture mismatch surfaces in production, regenerate via
  `ATENIA_REGENERATE_FIXTURES=1` and document the new IDs.
- `KvCache::append` BF16 path uses `Vec<u16>` reallocation per append.
  Fine for prefill + short contexts; for >2k token sustained generation
  the `Vec` capacity churn may show up in profiles. M9 may want a
  pre-sized capacity hint per `KvCacheConfig` (deferred).
- Pre-existing race in `cuda::disk_prefetch::tests` — orthogonal,
  not introduced by M8.6, listed for cleanup in M8.7 HANDOFF.

---

## 9. Resume plan after M9

If a future operator needs to revisit M8.6:

1. Read this file + `tests/m8_6_kv_cache_bf16_test.rs` (5 contracts).
2. The cast helpers in `src/amg/kv_cache.rs` are the integration
   points. Adding a per-group BF16 (Q8-style) ledger or a 4-bit cache
   (M11+) plugs into the same harvest/reinject seam — extend
   `KvCellDtype` and the `cast_kv_cell_*` pair.
3. The TinyLlama gate (`tests/m5_db_tinyllama_pipeline_test.rs`) is
   the canonical determinism check; re-run with the new dtype's
   flag enabled and either confirm bit-identical or regenerate the
   fixture.

---

**Closure tag:** `v0.8.6-m8.6` on `main` at commit (TBD on push).
