# MIXTRAL-DATA-PROVISION — PAUSED (safe state)

**Status: PAUSED, local state clean and resumable.** The Mixtral-8x7B-v0.1 weights
download (MIXTRAL-DATA-PROVISION) was **stopped** after repeated network stalls. The
local folder was left in a clean, partially-provisioned, safe-to-resume state. **No
forward, no certification, no runtime/loader/numerics change.** Weights are **not**
committed to git.

## What was attempted

`huggingface_hub.snapshot_download("mistralai/Mixtral-8x7B-v0.1",
local_dir="models/Mixtral-8x7B-v0.1", allow_patterns=["*.safetensors",
"model.safetensors.index.json"])` — i.e. the 19 safetensors shards + the index
(~93.4 GB), **excluding** `consolidated.*.pt` / `.bin` (per `MIXTRAL_DATA_READY.md`).

## Timeline / evidence of the failure

1. **Run 1** reached **~25.5 GB**, then the python process **died** (coincided with
   an environment hiccup). Nothing corrupt — hf `.incomplete` files are checksum-
   validated on resume.
2. **Run 2 (relaunch)** did **not** cleanly resume Run 1's 25.5 GB (hf restarted the
   in-flight shards). It progressed to **4 complete shards** but then **stalled**:
   directory size **frozen at the same byte count over a 20 s probe** (0 MB delta),
   python alive but **CPU effectively idle** — the classic network-stall signature
   (same as the earlier internet-cut on DeepSeek-V2-Lite). Not an Atenia bug.
3. Per the rules, the milestone was switched to **PAUSE** rather than blind retry.

## What was stopped / verified / cleaned

- **Process stopped:** killed the download PID (`Stop-Process -Force`), confirmed no
  `*mixtral_dl*` python remains. An **unrelated** python (`ex08_audit.py`) was left
  untouched. No other model folders touched.
- **Shards verified (read-only, no forward):** the 4 present `.safetensors` are
  **genuinely complete** — each file's byte size **equals** `8 + header_len +
  max(tensor data_offset)` from its safetensors header (the definitive truncation
  test), headers parse, dtype **BF16**:

  | Shard | Bytes | Tensors | Status |
  |---|---|---|---|
  | `model-00001-of-00019` | 4,892,809,584 | 51 | COMPLETE |
  | `model-00005-of-00019` | 4,983,004,016 | 55 | COMPLETE |
  | `model-00006-of-00019` | 4,983,004,016 | 55 | COMPLETE |
  | `model-00007-of-00019` | 4,899,035,248 | 48 | COMPLETE |

- **Cleaned (only broken partials):** removed
  `models/Mixtral-8x7B-v0.1/.cache/huggingface/download/` (the in-flight
  `*.incomplete` partials + `.lock` + `.metadata` of the shards that did **not**
  finish). **Kept** `config.json`, `generation_config.json`, all tokenizer files,
  and the **4 valid complete shards**. No valid weight was deleted; no other model
  was touched.

## Final local state

```
models/Mixtral-8x7B-v0.1/
  config.json, generation_config.json, special_tokens_map.json,
  tokenizer.json, tokenizer.model, tokenizer_config.json      (present)
  model-00001-of-00019.safetensors   (COMPLETE, BF16)
  model-00005-of-00019.safetensors   (COMPLETE, BF16)
  model-00006-of-00019.safetensors   (COMPLETE, BF16)
  model-00007-of-00019.safetensors   (COMPLETE, BF16)
```

- **Valid shards: 4 / 19** (~18.4 GiB). **`.incomplete`: 0.** **`.cache`: removed.**
- **`model.safetensors.index.json`: NOT present** (must be fetched on resume).
- **No download process alive.** Config = real Mixtral-8x7B-v0.1 (32 layers, 8
  experts, top-2, hidden 4096, inter 14336, vocab 32000, bf16).
- Weights are git-ignored (not committed).

## Remaining to fetch on resume

**15 shards** (`00002, 00003, 00004, 00008..00019`) + **`model.safetensors.index.json`**
≈ **~75 GB**. The 4 present shards will be detected as already-present (final file +
correct size) and **skipped**.

## Robust resume plan (for the next attempt — needs a stable connection)

The ~93 GB download needs a link that holds ~40–50 min. To avoid the parallel-worker
stall pattern, prefer **serial, per-shard, verified** fetching over a single big
`snapshot_download`:

1. **Stable connection first.** Confirm the link is steady before starting (the two
   failures were network, not tooling).
2. **Serial, low concurrency:** `snapshot_download(..., max_workers=1)` (or `2`), or
   loop `hf_hub_download` **one shard at a time** (`model-000NN-of-00019.safetensors`),
   so a drop kills at most one shard, not eight.
3. **Verify each shard right after it lands** with the truncation test (file size ==
   `8 + header_len + max data_offset`); re-fetch only the failing shard.
4. **Direct log to a file** (`> mixtral_dl.log 2>&1`), **no `| grep | tail` pipe**
   (which buffers/loses the final verdict — learned in the C5 runs).
5. **Progress watcher** that polls directory growth and flags a stall (no growth for
   N checks) instead of silently hanging.
6. **On stall: stop, clean only the broken `.incomplete`, then resume** — never blind
   relaunch (it can restart in-flight shards from zero, as happened here).
7. Fetch `model.safetensors.index.json` and validate that every tensor in the index
   resolves to a present shard before declaring provisioned.
8. Only then write `docs/MIXTRAL_PROVISIONED.md` and proceed to MIXTRAL-CERT.

## Risks / notes

- The **binding risk is connection stability**, not disk (F: has ~690 GB free) or
  tooling (`huggingface_hub` 1.12.0 works).
- hf's `.incomplete` resume is **not fully reliable across an abrupt kill** (Run 2
  restarted in-flight shards); the per-shard verified loop above sidesteps this.
- Atenia's `MoeRuntime` already loads sharded classic Mixtral (topology L0 certified);
  once all 19 shards + index land and validate, MIXTRAL-CERT (C1/C2 → L1, C4 → L2,
  C5 disk-tier → L3) can proceed, reusing the Qwen/DeepSeek machinery.
- **License:** Apache-2.0 (`mistralai/Mixtral-8x7B-v0.1`), gated acceptance done.

## Readiness for MIXTRAL-CERT

**Not ready yet** — provisioning is **partial (4/19 shards, no index)**. Resume the
download (robust plan above) on a stable link, validate all 19 shards + index, then
MIXTRAL-CERT can start. The 4 valid shards already on disk are reusable (no re-fetch).
