# Mixtral-8x7B Data-Readiness — MIXTRAL-DATA-READY (verify + prepare, no download)

> **RESOLVED (historical).** The weights were later provisioned (87 GB, 19 shards,
> `docs/MIXTRAL_PROVISIONED.md`) and Mixtral-8x7B-v0.1 reached **MoE-certified L3
> (active-path-certified)** via MIXTRAL-CERT-1/2/3. The "Mixtral remains L0" wording
> below is point-in-time and superseded. See `docs/HANDOFF_MIXTRAL_CERT_C5.md`.

**Verification + preparation only — NO weights downloaded, no runtime/loader/
numerics/test change, no model execution.** Confirms whether the real
Mixtral-8x7B weights can be safely provisioned for a future MIXTRAL-CERT-2
(L1/L2/L3 per ADR-007). Nothing here was executed beyond read-only local
inspection and read-only HuggingFace **metadata** API calls (no weight bytes
fetched). The actual download command is **proposed, not run** — it must not run
without explicit authorization.

## FASE 1 — Local state (verified)

`models/Mixtral-8x7B-v0.1/` contains **config + tokenizer only — no weights**:

| File | Size | Note |
|---|---|---|
| `config.json` | 720 B | `MixtralForCausalLM`, bf16, 32 layers, 8 experts/top-2, GQA 32/8, no shared, no SWA |
| `generation_config.json` | 116 B | bos 1 / eos 2 |
| `special_tokens_map.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` | ~2.3 MB | tokenizer present |

- **No `*.safetensors`, no `*.bin`, no `model.safetensors.index.json`, no
  `*.incomplete`, no `.cache`.** Clean state — a fresh download cannot collide
  with or be confused by a partial.
- **Disk free on `F:`** = **813 GB** (total 2000 GB, used 1187 GB) — ample for a
  ~93 GB download.

## FASE 2 — Source (verified, read-only metadata)

- **Repo:** `mistralai/Mixtral-8x7B-v0.1` (HuggingFace).
- **License:** **apache-2.0**. **`gated: False`** — no license-acceptance wall, no
  auth required to download. (A cached HF token exists at
  `~/.cache/huggingface/token` but is not needed.)
- **Weights (what we need):** **19 safetensors shards**
  `model-00001-of-00019.safetensors` … `model-00019-of-00019.safetensors`
  (~4.89–4.98 GB each) = **93.41 GB**, plus `model.safetensors.index.json`
  (92,658 B). Total to fetch ≈ **93.4 GB**.
- **What we must NOT fetch (avoid duplication):** the repo also ships the original
  Mistral format — **8 × `consolidated.0X.pt` ≈ 97 GB** (raw `.pt`, redundant with
  the safetensors). Including these would nearly **double** the download to ~190 GB
  for no benefit. **There are no `.bin` files.**
- **Recommended tool:** `huggingface_hub.snapshot_download` (installed: 1.12.0;
  the `hf` / `huggingface-cli` CLIs are not on PATH here). Built-in **resume** and
  per-file **sha/etag integrity** verification.
- **Atenia support:** `MoeRuntime` already loads **sharded classic Mixtral**
  (MOE-PROD-1) with **disk-tier residency** (MOE-PROD-2) and is scale-certified
  on the Mixtral topology — so once the shards land, the loader path is ready
  (no loader change needed).

## FASE 3 — Download plan (PROPOSED — do not run without authorization)

**Target:** the existing `models/Mixtral-8x7B-v0.1/` (so config + tokenizer stay
in place; `allow_patterns` restricts the fetch to weights, leaving them
untouched). **Excludes** the `consolidated.*.pt` files.

Recommended (Python API — most controllable, resumable, integrity-checked):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mistralai/Mixtral-8x7B-v0.1",
    local_dir="models/Mixtral-8x7B-v0.1",
    allow_patterns=["*.safetensors", "model.safetensors.index.json"],  # NOT consolidated.*.pt
    resume_download=True,        # safe to re-run; continues partial shards
    max_workers=4,
)
print("done")
PY
```

CLI alternative (if `hf` becomes available):

```bash
hf download mistralai/Mixtral-8x7B-v0.1 \
  --include "*.safetensors" "model.safetensors.index.json" \
  --local-dir models/Mixtral-8x7B-v0.1
```

Plan properties:
- **Exact destination:** `models/Mixtral-8x7B-v0.1/` (alongside the existing
  config/tokenizer; no existing model is moved or deleted).
- **Resume:** `resume_download=True` — interrupting and re-running continues from
  partial `.incomplete` files; no corruption.
- **Integrity:** `huggingface_hub` verifies each file's expected hash/etag on
  completion; a mismatched shard is re-fetched.
- **No duplication:** `allow_patterns` fetches only the 19 safetensors + index;
  the ~97 GB `consolidated.*.pt` are never downloaded; config/tokenizer already
  present are left as-is.
- **Safe abort:** Ctrl-C is safe (partial shard is resumable). To fully roll back,
  delete only the newly created `model-*.safetensors` + `model.safetensors.index.json`
  (NOT the config/tokenizer) — but **no deletion is part of this milestone**.
- **Expected footprint after download:** ~93.4 GB added → ~813 GB → ~720 GB free.

## FASE 4 — Post-download checks (PROPOSED, read-only — no model execution)

After a future authorized download, before any certification:

1. **All shards present:** 19 files `model-00001-of-00019.safetensors` …
   `model-00019-of-00019.safetensors` + `model.safetensors.index.json`.
2. **Index valid:** parse `model.safetensors.index.json`; every `weight_map`
   target resolves to a present shard; expected tensors (e.g.
   `model.layers.0.block_sparse_moe.experts.0.w1.weight`) are mapped.
3. **Total size:** ≈ 93.4 GB; each shard ~4.9 GB (no truncated/0-byte file).
4. **Header-only sanity (read-only):** open each shard's safetensors header (no
   tensor materialisation) to confirm dtype `bf16` and shapes match the config
   (hidden 4096, expert d_ff 14336, 8 experts/layer, 32 layers). This is the
   `atenia diagnose`-equivalent — metadata only.
5. **NO forward / NO generation / NO certification** in this readiness step — the
   model is not run. Certification is MIXTRAL-CERT-2+ (separate milestones).

## FASE 5 — Status, risks, next steps

### Current status
- **Mixtral is data-READY-to-provision but NOT provisioned.** Source verified
  (ungated, apache-2.0, 93.4 GB safetensors, index present, loader supported);
  local state clean (config+tokenizer only); disk ample (813 GB free). The only
  missing thing is the (authorized) ~93.4 GB download.
- Mixtral remains **MoE-certified L0** (C4 `mixtral_scale` 1.639e-7); L1/L2/L3 are
  blocked solely on these weights (per `docs/MIXTRAL_CERT_ROADMAP.md`).

### Risks
- **Download size/time (MEDIUM).** 93.4 GB, network-bound (tens of minutes to
  hours depending on link). Resumable, so interruption is low-impact.
- **Disk (LOW).** 93.4 GB into 813 GB free — comfortable; verify no other large
  job consumes the disk concurrently.
- **Wrong-artifact duplication (LOW, mitigated).** Without `allow_patterns` the
  fetch would also pull ~97 GB of `consolidated.*.pt` — the proposed command
  excludes them.
- **Integrity (LOW, mitigated).** `huggingface_hub` hash-verifies shards; the
  FASE 4 index/size/header checks add a second, read-only confirmation.
- **C5 RAM later (MEDIUM, out of scope here).** Certification (not this milestone)
  will face the ~11 GB/layer F64 reference cost noted in
  `docs/MIXTRAL_CERT_ROADMAP.md` — flagged for MIXTRAL-CERT-4, not for the download.

### Next steps toward MIXTRAL-CERT-2
1. **Get authorization** to download (93.4 GB) — this milestone stops here.
2. Run the FASE 3 command (safetensors + index only).
3. Run the FASE 4 read-only checks (shards/index/size/header).
4. Proceed to **MIXTRAL-CERT-2** (C1+C2 on real weights → L1), then **-3** (fold
   C4 → L2), then **-4** (C5 active-path → L3), reusing ~80–85 % of the Qwen-MoE
   tooling per `docs/MIXTRAL_CERT_ROADMAP.md`.

## Answers

- **Ready to download?** **Yes — verified and safe to provision**, pending only an
  explicit go-ahead. Source is ungated (apache-2.0), 19 safetensors shards + index
  exist, local state is clean, disk is ample, the loader already supports sharded
  Mixtral. **Not downloaded** in this milestone.
- **Space required:** **~93.4 GB** (safetensors + index); leaves ~720 GB free.
  Do **not** also fetch the ~97 GB `consolidated.*.pt`.
- **Recommended command:** the `snapshot_download(... allow_patterns=["*.safetensors",
  "model.safetensors.index.json"], resume_download=True)` in FASE 3.

*Verification + documentation only — no weights downloaded, no source/runtime/
loader/numerics/test change, no model executed.*
