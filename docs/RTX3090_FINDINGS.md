# RTX 3090 Findings - manual_20260511_132534

Source logs:

```text
bench_logs/manual_20260511_132534
```

## Server Fingerprint

- GPU: NVIDIA GeForce RTX 3090
- Driver: 580.126.09
- CUDA reported by driver: 13.0
- RAM: 91 GiB usable
- CPU path: AVX2=true, AVX512=true, FMA=true
- Rust/Cargo: rustc 1.95.0, cargo 1.95.0
- Build: release build completed successfully

## Repo Smokes

The Rust test surface is green on the 3090 server:

| Gate | Result |
| --- | --- |
| `cargo test --lib` | 321 passed |
| `cargo test --lib gguf_` | 30 passed |
| TinyLlama end-to-end | passed |
| SmolLM2 end-to-end | passed |
| Qwen 2.5 1.5B end-to-end | passed |
| Llama 3.2 1B end-to-end | passed |
| TinyLlama BF16 storage | passed |
| TinyLlama GPU matmul | passed |
| TinyLlama disk spill | passed |
| BF16 full-family validation | 4 passed |
| M4.7 full-family validation | 4 passed |
| M8.5 full-family validation | 4 passed |
| M11.D GGUF diagnostics | 13 passed |

This means the server, fixtures, and local CUDA-capable test surface are usable.
The main problems are in CLI loading / residency / dispatch behavior.

## CLI Outcomes

| Case | Result | tok/s | Notes |
| --- | ---: | ---: | --- |
| GGUF TinyLlama Q4 | OK | 3.15 | quantized manifest |
| GGUF TinyLlama Q8 | OK | 3.11 | quantized manifest |
| GGUF Llama 3.2 1B Q4 | OK | 0.35 | quantized manifest |
| GGUF SmolLM2 Q4 | OK | 0.60 | quantized manifest |
| GGUF Phi 3.5 Q4 | FAIL | - | missing/underived LongRope metadata |
| TinyLlama safetensors manifest | OK | 3.16 | certified |
| TinyLlama safetensors fast | OK | 3.11 | fast override |
| SmolLM2 manifest | OK | 0.60 | certified |
| SmolLM2 fast | OK | 0.60 | fast override, no material speedup |
| Qwen 2.5 1.5B manifest | OK | 0.42 | manifest already recommends fast |
| Qwen 2.5 1.5B fast | OK | 0.43 | same practical path |
| Llama 3.2 1B manifest | OK | 0.35 | certified |
| Llama 3.2 1B fast | OK | 0.35 | fast override, no material speedup |
| Phi 3.5 manifest | OK | 2.13 | certified |
| Phi 3.5 fast | OK | 2.00 | fast slightly slower in this run |
| Mistral 7B manifest | FAIL | - | BF16->VRAM slow-path upload |
| Mistral 7B fast | OK | 1.42 | fast avoids failure |
| Gemma 2 2B manifest | OK | 0.11 | severe underutilization |
| Gemma 2 2B fast | OK | 0.11 | 100-token rerun confirms not short-run overhead |
| Falcon 3 7B manifest | FAIL | - | BF16->VRAM slow-path upload |
| Falcon 3 7B fast | OK | 0.63 | fast avoids failure |
| Llama 2 13B manifest | OK | 0.13 | large resident plan |
| Llama 2 13B fast | OK | 0.38 | ~2.9x faster than manifest |

## GPU Utilization Signal

The `nvidia-smi dmon` logs show stable VRAM allocation but low average SM
utilization. This points away from dynamic spilling and toward intermittent
kernel launch / CPU orchestration / synchronization / dispatch inefficiency.

Examples:

| Case | tok/s | SM avg | SM max | Samples <=10% SM | Samples >=50% SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma2 fast 100 tok | 0.115 | 8.1% | 92% | high | rare |
| Gemma2 manifest | 0.112 | 7.5% | 88% | 95.9% | 2.1% |
| Mistral fast 100 tok | 1.413 | 19.9% | 96% | much lower than Gemma | better |
| Llama2 13B manifest | 0.134 | 21.5% | 58% | 53.5% | 10.6% |
| TinyLlama manifest | 3.156 | 14.9% | 37% | 56.2% | 0% |

Important nuance: the 1-second `dmon` sample did not usually catch literal 0%
SM, but most cases spend a large majority of samples at <=10% SM. The user's
observation of brief drops to zero is consistent with this sawtooth behavior
being visible in a different monitor or sampling cadence.

## Residency / Size Inconsistencies

Several logs mix source model size and resident plan size without making the
distinction explicit.

Examples:

| Case | Source model log | VRAM plan | RAM plan | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Gemma2 manifest | 4.87 GiB | 7.54 GiB | 1.10 GiB | certified path expands many weights |
| Gemma2 fast | 4.87 GiB | 3.77 GiB | 1.10 GiB | fast halves VRAM but still leaves RAM |
| Phi3.5 manifest | 7.12 GiB | 13.50 GiB | 0.37 GiB | certified expansion |
| Phi3.5 fast | 7.12 GiB | 6.75 GiB | 0.37 GiB | fast halves VRAM |
| Mistral manifest | 13.50 GiB | 21.44 GiB | 2.78 GiB | certified expansion then upload failure |
| Mistral fast | 13.50 GiB | 13.00 GiB | 0.50 GiB | fast fits and runs |

This may be internally coherent if source BF16/F16 expands to F32 in certified
mode, but the log should say `source_size` and `resident_estimate` separately.

## Suspicious RAM Placement Despite Free VRAM

Even when the RTX 3090 has room, the planner leaves tensors in RAM:

- Gemma2 fast: 3.77 GiB VRAM + 1.10 GiB RAM
- Qwen 1.5B: 2.44 GiB VRAM + 0.43 GiB RAM
- TinyLlama fast: 1.80 GiB VRAM + 0.24 GiB RAM
- Llama 3.2 1B fast: 1.81 GiB VRAM + 0.49 GiB RAM
- Mistral fast: 13.00 GiB VRAM + 0.50 GiB RAM

The current logs do not explain why these tensors are RAM-resident. We need
per-tensor or per-category reason counts:

- not GPU eligible
- budget exceeded
- policy CPU/RAM
- dtype/resident-size expansion
- safety headroom
- architecture transform constraint

## Clear Bugs / Work Items

1. **BF16->VRAM slow-path upload failure**
   - `st_mistral7b_manifest` fails on `model.layers.26.mlp.gate_proj.weight`.
   - `st_falcon3_7b_manifest` fails on `model.layers.20.mlp.gate_proj.weight`.
   - The same models work in fast mode, so this is likely certified/slow-path
     upload or dtype handling rather than model corruption.

2. **Gemma2 dispatch / utilization**
   - 100-token fast run still only reaches ~0.115 tok/s.
   - SM avg remains ~8.1% with brief high spikes.
   - This is not just short prompt overhead.

3. **Planner explanation gap**
   - RAM placement with VRAM available needs reason logging.
   - Source size vs resident size needs explicit logging.

4. **Phi GGUF LongRope metadata**
   - Safetensors Phi works.
   - GGUF Phi fails because LongRope is missing or not derived from GGUF metadata.

5. **Fast mode does not universally improve speed**
   - SmolLM2, Llama 3.2 1B, Qwen, Gemma show little to no improvement.
   - Need dispatch counters by mode/path to prove which kernels actually run.

## Recommended Next Steps

1. Add structured planner diagnostics:
   - VRAM total/free/budget.
   - source bytes vs resident bytes.
   - per-tier reason counts.
   - top N RAM tensors with reason and dtype.

2. Fix or instrument BF16->VRAM slow-path upload:
   - Reproduce with Mistral/Falcon certified.
   - Log tensor shape, dtype, byte count, target tier, CUDA error, allocation size.

3. Add runtime dispatch counters to CLI output/stderr:
   - CPU matmul count.
   - CUDA F32/TF32 count.
   - CUDA BF16 native count.
   - CUDA BF16 slow-path count.
   - host/device transfer count and bytes.

4. Investigate Gemma2 execution:
   - Compare node/operator mix against Mistral/Phi.
   - Check SoftCap / GeGLU / dual-norm routing.
   - Confirm whether large ops hit CUDA or CPU.

5. Fix the small test/build portability issues separately:
   - `llama_3_2_end_to_end_test.rs` must match `LongRope`.
   - `build.rs` should not require manual `RUSTFLAGS="-L native=$(pwd)"`.
