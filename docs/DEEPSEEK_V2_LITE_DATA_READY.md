# DeepSeek-V2-Lite Data-Readiness — DEEPSEEK-V2-LITE-DATA-READY (verify + prepare, no download)

**Verification + preparation only — NO weights downloaded, no runtime/loader/
numerics/ATK change, no model executed.** Confirms whether the real
DeepSeek-V2-Lite weights can be safely provisioned for **MLA-1** (the ADR-007 cert
now unblocked by MLA-0). Only read-only local inspection + read-only HuggingFace
**metadata** API calls were performed (no weight bytes fetched). The download
command is **proposed, not run** — it must not run without explicit authorization.

## FASE 1 — Local state (verified)

- **No DeepSeek-V2-Lite present.** `models/` contains only the **R1 distills**
  (`DeepSeek-R1-Distill-Llama-8B-…GGUF`, `DeepSeek-R1-Distill-Qwen-7B-…GGUF`) —
  those are **Llama/Qwen dense** distills, **not** the MLA/MoE V2-Lite. No
  `DeepSeek-V2-Lite*` folder, no `*.safetensors`/`*.bin`/index, no partials, no
  `.cache`. **Clean state** — a fresh download cannot collide with anything.
- **Disk free on `F:`** = **812 GB** (total 2000 GB, used 1188 GB) — ample for a
  ~31 GB download.

## FASE 2 — Source (verified, read-only metadata)

- **Repo:** `deepseek-ai/DeepSeek-V2-Lite` (HuggingFace).
- **License:** **`other`** — the **DeepSeek License Agreement** (a custom license,
  not Apache/MIT). **`gated: False`** → downloadable without an auth wall, **but**
  the DeepSeek License terms (use restrictions, e.g. acceptable-use) **should be
  reviewed and accepted before download** — this is the one non-trivial caveat.
- **Weights (what we need):** **4 safetensors shards**
  `model-00001-of-000004.safetensors … model-00004-of-000004.safetensors` =
  **31.41 GB**, plus `model.safetensors.index.json` (479,924 B). Total weights to
  fetch ≈ **31.4 GB**.
- **No redundant weight formats.** Unlike Mixtral (which shipped ~97 GB of extra
  `consolidated.*.pt`), V2-Lite has **no `.bin` and no `.pt`** — only safetensors.
  Nothing large to exclude.
- **Non-weight files (all tiny):** `config.json` (1.5 KB), `generation_config.json`,
  `tokenizer.json` (4.6 MB), `tokenizer_config.json`, `tokenization_deepseek_fast.py`,
  `modeling_deepseek.py` (79 KB), `configuration_deepseek.py` (10 KB), `LICENSE`,
  `README.md`. The `modeling_deepseek.py`/`configuration_deepseek.py` are the
  HF **remote-code** path; **Atenia does not need them** (it has its own MLA
  forward, and the MLA-1 reference uses transformers' **native** `DeepseekV2`
  class — the same one MLA-0 validated against). They are tiny, so including them
  is harmless but optional.
- **No token/auth required** (`gated: False`).
- **Atenia support:** MLA-0 closed the V2-Lite prerequisites (YaRN, dense-first,
  routing convention); the experimental `MoeRuntime` MLA path + `build_deepseek`
  parse `rope_scaling`/`first_k_dense_replace`/`norm_topk_prob`, and the sharded
  loader handles `model.safetensors.index.json`. So once the shards land, the
  load + forward path is ready for MLA-1 (cert, not productive serving).

## FASE 3 — Download plan (PROPOSED — do not run without authorization)

**Target:** a fresh `models/DeepSeek-V2-Lite/`. Fetches the 4 safetensors + index
+ config + tokenizer (+ the tiny `.py`/LICENSE). There is no redundant large
artifact to ignore.

Recommended (Python API — resumable, integrity-checked):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-V2-Lite",
    local_dir="models/DeepSeek-V2-Lite",
    allow_patterns=[
        "*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "*.py",        # tiny remote-code files (optional; Atenia uses native DeepseekV2)
        "LICENSE",
    ],
    resume_download=True,   # safe to re-run; continues partial shards
    max_workers=4,
)
print("done")
PY
```

Plan properties:
- **`repo_id`:** `deepseek-ai/DeepSeek-V2-Lite`.
- **`local_dir`:** `models/DeepSeek-V2-Lite/` (new; no existing model moved/deleted).
- **`allow_patterns`:** weights + index + config + tokenizer (+ tiny `.py`/LICENSE).
- **`ignore_patterns`:** none needed (no `.bin`/`.pt`/consolidated redundancy in
  this repo).
- **Resume:** `resume_download=True` — interrupt + re-run continues partial shards.
- **Integrity:** `huggingface_hub` verifies each file's hash/etag on completion; a
  mismatched shard is re-fetched. FASE 4 adds a second read-only confirmation.
- **`max_workers`:** 4 (reasonable for 4 shards on NVMe).
- **Footprint after download:** ~31.4 GB added → ~812 GB → ~780 GB free.
- **Safe abort:** Ctrl-C (partial shard resumable). No deletion is part of this
  milestone.

## FASE 4 — Post-download checks (PROPOSED, read-only — no forward/generate/cert)

After a future authorized download, before MLA-1:

1. **All shards present:** `model-00001-of-000004.safetensors` …
   `model-00004-of-000004.safetensors` + `model.safetensors.index.json`.
2. **Index valid:** parse `model.safetensors.index.json`; every `weight_map`
   target resolves to a present shard.
3. **Total size:** ≈ 31.4 GB; each shard non-zero/non-truncated.
4. **Header-only sanity (read-only, no tensor materialisation):** open each shard's
   safetensors header and confirm **bf16** + the expected MLA tensor names
   (`self_attn.kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `q_proj`,
   `o_proj`), **layer 0 dense** (`model.layers.0.mlp.{gate,up,down}_proj`, no
   experts) and **layers 1.. MoE** (`mlp.gate`, `mlp.experts.*`, `mlp.shared_experts.*`).
5. **Config matches expected (V2-Lite):** `q_lora_rank=null`, `kv_lora_rank=512`,
   `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`,
   `n_routed_experts=64`, `num_experts_per_tok=6`, `n_shared_experts=2`,
   `first_k_dense_replace=1`, `norm_topk_prob=false`, `rope_scaling.type=yarn`
   (factor 40), 27 layers, hidden 2048, 16 heads.
6. **No redundant files:** confirm no `.bin`/`.pt`/`consolidated.*` were fetched.
7. **NO forward / NO generate / NO certification** in this readiness step. (The MLA
   load + forward correctness is already validated by MLA-0 on a tiny fixture; the
   real-weight cert is MLA-1.)

## FASE 5 — Status, risks, next steps

### Current status
- **DeepSeek-V2-Lite is data-READY-to-provision but NOT provisioned.** Source
  verified (ungated, 31.41 GB safetensors, index present, no redundant artifacts,
  native `DeepseekV2` supported); local state clean; disk ample (812 GB free).
  Atenia's MLA path is MLA-0-ready. The only missing thing is the (authorized)
  ~31.4 GB download.

### Risks
- **License (MEDIUM — review before download).** `other` = DeepSeek License
  Agreement, not a permissive OSS license; not gated, but its terms should be
  reviewed/accepted. This is the one item needing a human decision.
- **Download size/time (LOW–MEDIUM).** 31.4 GB, network-bound; resumable.
- **Disk (LOW).** 31.4 GB into 812 GB free.
- **Remote-code confusion (LOW, mitigated).** The repo ships `modeling_deepseek.py`;
  Atenia and the MLA-1 reference use transformers' **native** `DeepseekV2`, so
  `trust_remote_code` is not required. The `.py` files are tiny and optional.
- **C5 RAM later (LOW — favourable).** Unlike Mixtral, V2-Lite is **Qwen-scale**
  (experts `moe_intermediate=1408`, Qwen-sized) → L1/L2/L3 are RAM-feasible on the
  32 GB host (per `DEEPSEEK_V2_LITE_FEASIBILITY`).

### Next steps toward MLA-1
1. **Review + accept the DeepSeek License**, then **authorize** the ~31.4 GB download.
2. Run the FASE 3 command (safetensors + index + config + tokenizer).
3. Run the FASE 4 read-only checks (shards/index/size/header/config).
4. Proceed to **MLA-1**: ADR-007 C1–C5 on real V2-Lite (reuse ~70–80% of the Qwen
   tooling + the MLA-0 forward) → L1 → L2 → L3.

## Answers

- **Ready to download?** **Yes — verified and safe to provision**, pending (a) a
  review/acceptance of the **DeepSeek License** and (b) explicit go-ahead. Source
  is ungated, 4 safetensors shards + index exist, local state clean, disk ample,
  Atenia MLA-0-ready. **Not downloaded** in this milestone.
- **Repo:** `deepseek-ai/DeepSeek-V2-Lite`.
- **Expected size:** **~31.4 GB** (4 safetensors shards + index; + ~5 MB small
  files). Leaves ~780 GB free.
- **Required files:** `model-0000{1..4}-of-000004.safetensors`,
  `model.safetensors.index.json`, `config.json`, `generation_config.json`,
  `tokenizer.json`, `tokenizer_config.json`.
- **Files to exclude:** none mandatory (no redundant `.bin`/`.pt`); the remote-code
  `*.py` are optional (Atenia uses native `DeepseekV2`).
- **Proposed command:** the `snapshot_download(... allow_patterns=[...],
  resume_download=True, max_workers=4)` in FASE 3.

*Verification + documentation only — no weights downloaded, no source/runtime/
loader/numerics/ATK change, no model executed.*
