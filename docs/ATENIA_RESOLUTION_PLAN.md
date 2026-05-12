# ATENIA Resolution Plan - post RTX 3090 battery

Scope: follow-up to `bench_logs/manual_20260511_132534` and
`docs/RTX3090_FINDINGS.md`.

The server run proved that the repo smoke surface is healthy, but the CLI path
still has concrete residency, upload, and dispatch problems. This plan is
intentionally incremental: fix one layer, run a small battery, then continue.

## Current Diagnosis

### 1. RAM placement with free VRAM is mostly current policy

`src/gpu/tier_plan.rs` only marks tensors as GPU-eligible when:

```text
rank >= 2 && name.ends_with("_proj.weight")
```

Everything else is forced to RAM or disk even if the RTX 3090 has free VRAM.
That includes embeddings, tied LM-head inputs, norms, biases, masks, and
architecture-specific small tensors.

Reason: the executor still has many CPU-only nodes that call `ensure_cpu`
before reading tensor data. Keeping those tensors in VRAM today would force
downloads or hit unsupported storage paths.

Concrete impact:

- The planner is not trying to fill VRAM.
- The logs do not explain this policy, so it looks like a bad capacity
  calculation.
- Gemma2/Qwen/Llama 3.2 keep visible RAM allocations even when VRAM has room.

### 2. Source model size and resident size are mixed in the logs

The loader logs `model_total_bytes` in source/storage units, then logs the tier
plan in resident units. Certified mode often stores VRAM weights as F32
resident tensors (`numel * 4`), while fast mode stores BF16 resident tensors
(`numel * 2`).

That explains the apparent doubling:

- Gemma2 source: 4.87 GiB
- Gemma2 certified VRAM plan: 7.54 GiB
- Gemma2 fast VRAM plan: 3.77 GiB

This can be coherent, but the current log format hides the distinction.

### 3. Manifest failures are concentrated in BF16 to F32 VRAM slow path

Mistral 7B and Falcon 3 7B fail in manifest/certified mode with:

```text
BF16->VRAM slow-path upload failed for ...
```

The same checkpoints pass in fast mode. That points to the certified slow path:

```text
Vec<f32> transforms -> Vec<u16> -> bf16_to_f32_resident_in_vram(...)
```

The failure may be allocation, cudaMemcpy, or the BF16-to-F32 kernel launch,
but the helper currently returns only `None`, so the real CUDA error is lost.

### 4. Low GPU utilization is expected from the current execution shape

The 3090 SMI logs show stable VRAM but low average SM utilization. This is not
dynamic spilling. The code path explains the sawtooth:

- Generation rebuilds a fresh decode graph for every token.
- Many nodes are CPU-only: `RmsNorm`, `Reshape`, `Permute`, `Transpose2D`,
  `IndexSelect`, `Softmax`, `SoftCap`, `BroadcastAdd`, `BroadcastMul`,
  `Concat`, activations.
- Projection matmuls can hit CUDA, but outputs usually return to CPU between
  ops.
- The tied LM head path transposes `embed_tokens.weight` every forward. The
  builder comment calls this out as a known follow-up; for large vocab models
  this can be hundreds of MiB of CPU work per step.

This explains why VRAM stays stable while SM utilization spikes and drops.

### 5. Gemma2 is a worst-case path

Gemma2 adds dual norms, GeGLU, attention/logit SoftCap, embedding scale, and a
large tied-vocab LM head. In the current executor those extra ops are CPU-heavy,
so fast BF16 matmul does not move the needle.

The 100-token rerun confirms this is structural, not short-run overhead.

## Resolution Plan

### Pass 1 - make the engine tell the truth

Goal: improve logs without changing behavior.

Changes:

- Extend tier-plan logging with:
  - source bytes vs resident bytes
  - effective kernel dtype
  - VRAM total/free/budget/headroom
  - RAM headroom and post-VRAM RAM pressure
  - per-tier counts grouped by reason
- Add a planner diagnostic mode, for example `ATENIA_PLAN_TRACE=1`, that prints
  the top RAM tensors with reason:
  - `not_gpu_eligible`
  - `vram_budget_exceeded`
  - `ram_headroom_reserved`
  - `disk_staging_reserved`
- Add CLI dispatch counters to stderr or JSON:
  - resident GPU matmuls
  - roundtrip GPU matmuls
  - non-pooled GPU matmuls
  - legacy GPU matmuls
  - BF16 certified vs BF16 native/fast counts
  - loader VRAM fast/slow/BF16/INT8 counters

Validation:

- Run one small model and one problematic model:
  - TinyLlama safetensors manifest, 20 tokens
  - Gemma2 fast, 20 tokens
- Expected result: same generated behavior, but logs explain why tensors remain
  in RAM and which dispatch paths actually ran.

Status:

- Pass 1 landed: tier-plan reason tracing and CLI counters are available via
  `ATENIA_PLAN_TRACE=1`.
- Pass 1.1 landed: node timing summaries are available via
  `ATENIA_NODE_TIMING=1`.
- Pass 1.2 landed: tied LM heads now use `MatMulRhsTransposed` instead of
  materialising `Transpose2D(embed_tokens.weight)` per forward.
- Pass 1.3 landed: `IndexSelect` over `CpuBf16` / `CpuBf16Shared` embedding
  tables decodes only selected rows instead of upcasting the full table.

### Pass 2 - fix the BF16->VRAM slow-path failure

Goal: make manifest/certified Mistral and Falcon either load or fail with the
exact CUDA reason.

Changes:

- Replace `Option<TensorGPU>` upload helpers with an internal error detail or
  add a debug variant used by the loader.
- Log on failure:
  - tensor name
  - shape
  - numel
  - BF16 bytes
  - F32 resident bytes requested
  - free VRAM before upload if available
  - CUDA failing operation: alloc, HtoD, kernel launch
- Add a targeted test/helper that uploads the failing tensor shape without
  loading the full model.

Validation:

- Re-run:
  - Mistral 7B manifest, 20 tokens
  - Falcon 3 7B manifest, 20 tokens
- Expected result: either both load, or we get a precise CUDA failure instead
  of `InvalidFormat("... upload failed ...")`.

### Pass 3 - remove the remaining CPU/GPU roundtrip tax

Goal: improve utilization after Pass 1 removed the tied-LM-head and embedding
decode bottlenecks.

Original first target: tied LM head.

Current path:

```text
embed_tokens.weight -> Transpose2D on CPU every forward -> MatMul
```

Changes:

- Avoid materializing the transposed tied embedding on every token.
- Options to evaluate:
  - add a transpose-free matmul mode for `x @ embed_tokens.T`
  - cache the transposed LM head once in `WeightStore`
  - add `model.embed_tokens.weight` / tied LM-head GPU residency only after the
    executor can consume it without forced CPU download

Validation:

- Rerun Qwen 1.5B, Llama 3.2 1B, Gemma2 2B with 100 tokens.
- Expected result: Gemma2/Qwen should improve more than TinyLlama because their
  vocabularies are large.

Status:

- Original tied-head target already landed during Pass 1:
  - Pass 1.2 replaced per-forward `Transpose2D(embed_tokens.weight)` with
    `MatMulRhsTransposed`.
  - Pass 1.3 made `IndexSelect` decode only selected BF16 embedding rows.
- Pass 3.1 adds `ATENIA_MATMUL_TRACE=1`, an opt-in MatMul trace that records:
  - branch: `resident_or_mixed`, `roundtrip`, `non_pooled`, `legacy_gpu`,
    `cpu_ato`, or `cpu_fallback`
  - RHS storage tier before dispatch
  - RHS parameter name when the graph builder knows it
  - MatMul shape and elapsed time
- The CLI prints a top-40 MatMul trace summary when the flag is enabled.
- RTX 3090 Falcon trace showed the remaining cost is not random spilling:
  `lm_head.weight` and late-layer MLP projections were the top CPU
  `non_pooled` MatMuls.
- Pass 3.2 starts the fix in the tier planner:
  - untied `lm_head.weight` is GPU-eligible because MatMul can now consume it
    directly;
  - VRAM packing remains layer-local for projection weights, with `lm_head`
    considered after projections so large vocab heads do not evict many
    resident layer weights;
  - embeddings, norms, biases, and tied embedding tables remain CPU/RAM unless
    a supported GPU consumer exists.
- A first pure-hotness ordering was tested on Falcon and rejected: it removed
  `lm_head` from the top CPU list but dropped resident matmuls too much and
  inflated `roundtrip` dispatches.
- Pass 3.3 adds an opt-in `ATENIA_MATMUL_WEIGHT_CACHE=1` experiment for
  inference-only `lm_head.weight`: if post-load VRAM has room, Atenia uploads
  the CPU LM head once and reuses it through the mixed-resident MatMul path.
  If allocation fails, execution falls back to the normal CPU/non-pooled path.

Next validation:

- On RTX 3090, rerun a short certified Falcon probe with:

```text
ATENIA_MATMUL_TRACE=1 ATENIA_GPU_FALLBACK_TRACE=1 atenia generate ...
```

- Expected result with `ATENIA_MATMUL_WEIGHT_CACHE=1`: `lm_head.weight` should
  move from `non_pooled b_storage=Cpu` to `resident_or_mixed b_storage=Cuda`
  when spare VRAM exists. Remaining CPU entries should guide the next choice
  between layer-block packing and a broader transient/LRU VRAM cache.

### Pass 4 - Gemma2-specific CPU-only ops

Goal: make Gemma2 stop being the pathological case.

Changes:

- Instrument node timing per op type for a Gemma2 20-token run.
- Based on timing, port only the hot ops first:
  - likely `SoftCap`
  - `GELU/GeGLU` elementwise section
  - `RmsNorm`
  - `Softmax` only if the timing says it matters at decode length

Validation:

- Compare Gemma2 manifest/fast 100-token tok/s and SMI SM avg.
- Expected result: fewer long CPU gaps and higher average SM utilization.

### Pass 5 - expand GPU residency policy carefully

Goal: use more VRAM only after consumers are ready.

Changes:

- Do not simply mark every 2D tensor GPU-eligible.
- Add eligibility by operator support:
  - projection weights: already supported
  - untied `lm_head.weight`: after LM-head GPU path exists
  - tied embeddings: after transpose-free or cached-transpose path exists
  - norms/biases: after GPU kernels exist or CPU download is acceptable and
    measured
- Keep ADR-004/ADR-005 split:
  - certified path must preserve strict drift
  - fast path may use native BF16 only under manifest/override policy

Validation:

- Compare planner traces before/after.
- Verify no regression in BF16 full-family, M4.7 family, M8.5 family, and
  M11.D diagnostics.

### Pass 6 - portability cleanup

Goal: make the server workflow reproducible without manual linker hacks.

Changes:

- Fix `build.rs` so CUDA static libs are produced in and linked from a stable
  build output location.
- Link every generated CUDA static lib on Linux and Windows.
- Remove root-level generated `lib*.a` artifacts from the normal workflow.

Validation:

- Fresh clone/copy on Ubuntu:
  - `CC=gcc-13 CXX=g++-13 cargo build --release --bin atenia`
  - no manual `RUSTFLAGS="-L native=$(pwd)"`

## Recommended Immediate Next Move

Start with Pass 1.

It is low risk, ADR-safe, and gives us the evidence needed to avoid guessing.
After Pass 1, a short RTX 3090 rerun should tell us whether to attack the
BF16 upload failure first or the tied LM-head bottleneck first.

## Architecture Detour - Internal Model Adapter Layer

Before adding more model-family features to the core, freeze current behavior
behind an internal adapter layer. This is not a public SDK yet; the trait can
still change while the design settles.

Internal law:

```text
Atenia Core executes; adapters describe models.
```

Initial order:

- Keep current Llama / Qwen2 / Mistral / Phi3 / Gemma2 behavior intact.
- Introduce a small internal registry plus modular traits for graph building,
  HF mapping, GGUF mapping, and future residency hints.
- Migrate Llama-family routing first because it is the baseline path.
- Keep Phi3 and Gemma2 as thin adapters around their existing specialized
  builders and mappers.
- Split Qwen2, Mistral, and Falcon3 only after the base layer proves stable.
- Consider a public Atenia Adapter SDK only after the internal contract has
  survived several model-family migrations.

First-pass acceptance rule: same behavior, same tests passing, no required
performance gain.

Phase 2 keeps the same rule and adds internal adapter metadata:

- stable model-family identity separate from raw HuggingFace architecture
  strings;
- capability flags for family-specific features such as Phi3 fused QKV /
  gate-up mapping and Gemma2 softcaps;
- descriptive residency hints that mirror today's tier policy, but do not yet
  drive the planner.

Planner integration is intentionally deferred until the hints can be validated
against RTX 3090 traces family by family.

## Future Research Note - Long Context Governor

Do not fold this into Pass 3 / Pass 4. The current passes are about making the
existing exact Llama-family execution path faster and more honest.

There is a separate future track worth preserving: an optional Atenia Long
Context Governor inspired by subquadratic / sparse-attention systems. It should
remain a layer on top of the certified core, not a replacement for it.

Possible staged path:

- Paged KV cache and KV block residency across VRAM / RAM / disk. This should
  not change model math.
- Content-aware KV block priority for long sessions: recent blocks, system
  prompt, referenced code/doc spans, and attention-hot blocks stay closer to
  VRAM.
- Experimental selective attention over KV blocks, behind an explicit opt-in
  such as `ATENIA_EXPERIMENTAL_SPARSE_ATTENTION=1`. This changes the math and
  must not claim ADR-004 strict equivalence by default.
- Native support for future sparse / linear-attention model families if open
  checkpoints appear, rather than pretending existing dense-attention
  checkpoints can be converted by runtime alone.

Acceptance rule: exact execution remains the default; experimental long-context
execution must be measurable, reversible, clearly logged, and separately
documented.
