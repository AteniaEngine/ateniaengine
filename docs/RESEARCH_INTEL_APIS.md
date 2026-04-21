# Research: Multi-vendor GPU APIs for Atenia Engine

Status: Investigation complete — no integration work yet.
Relevant milestone: APX v22+ (multi-backend foundation).

## Context

Atenia's third design pillar is vendor-neutrality: the engine should
coordinate hardware across NVIDIA, AMD, Intel and Apple, not lock into
a single vendor. Today the engine uses `nvidia-smi` to probe NVIDIA
VRAM and `sysinfo` to probe system RAM. To extend to other vendors,
an abstraction layer is needed on top of vendor-specific or
cross-vendor APIs.

This document captures the investigation into which API(s) could
underpin such an abstraction, performed on a host with:

- NVIDIA GeForce RTX 4070 Laptop GPU (dedicated, 7948 MB VRAM)
- Intel UHD Graphics (integrated, 128 MB dedicated + 16 GB shared)
- Windows 11, Intel Core i7-14650HX

## APIs evaluated

### OpenCL

- **Status on host:** ICD Loader present (`C:\Windows\System32\OpenCL.dll`),
  but the Khronos OpenCL Vendors registry key was empty.
- **Implication:** OpenCL runtime is installed at the OS level, but no
  vendor (Intel or NVIDIA) has registered an implementation visible to
  the loader. OpenCL calls would fail with no platforms available.
- **To enable:** install Intel OpenCL Runtime or include the OpenCL
  package that ships with CUDA Toolkit.
- **Cross-vendor:** yes (Intel, AMD, NVIDIA, Apple historically).
- **Maturity in Rust:** good (`opencl3`, `ocl` crates are mature).
- **Verdict:** viable in principle, but requires installation to
  activate on this host. Deferred.

### Level Zero (Intel native)

- **Status on host:** not installed. Requires Intel oneAPI Base Toolkit
  (~6-8 GB) or at minimum the Level Zero Loader runtime.
- **Cross-vendor:** no (Intel-specific, though AMD has announced
  experimental support).
- **Maturity in Rust:** limited (`level-zero-sys` is lower-level bindings).
- **Verdict:** most powerful for Intel-specific queries, but scope is
  narrow and installation is heavy. Not a first candidate.

### DirectML / DXGI (Microsoft, Windows-only)

- **Status on host:** present out of the box. `C:\Windows\System32\DirectML.dll`
  is a 10 MB runtime shipped with Windows.
- **DXGI (adapter enumeration):** confirmed working. `dxdiag` lists
  both GPUs with card name, chip type, dedicated and shared memory.
  A single API call on this host returns:

  ```
  Adapter 0: Intel UHD Graphics
    Dedicated Memory: 128 MB
    Shared Memory:    16235 MB

  Adapter 1: NVIDIA GeForce RTX 4070 Laptop
    Dedicated Memory: 7948 MB
    Shared Memory:    16235 MB
  ```

- **Cross-vendor:** yes on Windows — DXGI enumerates any GPU that
  supports DirectX 12, which covers Intel, NVIDIA, AMD, Qualcomm.
- **Maturity in Rust:** good (`windows` crate provides bindings).
- **Platform scope:** Windows only. Does not help for Linux or macOS
  deployments.
- **Verdict:** strongest candidate for a first multi-vendor probe on
  Windows. Already usable without any installation.

## Key semantic findings

### Shared memory is the same RAM

On this host, both GPUs report "Shared Memory: 16235 MB". This is the
same physical system RAM that both GPUs are allowed to borrow. When
Atenia counts available memory across tiers, it must not sum VRAM +
shared memory of each adapter as if they were independent pools. The
correct mental model is:

- NVIDIA dedicated: 7948 MB (only for NVIDIA)
- Intel dedicated: 128 MB (negligible; useful as a signal, not capacity)
- System RAM: 16235 MB (shared pool both GPUs and the CPU draw from)

Total addressable for execution is approximately 24 GB, not the 40 GB
that naive summation would suggest. This is the kind of semantic
subtlety that makes honest multi-tier coordination difficult, and
which a vendor-neutral abstraction must get right.

### Dynamic vs static memory

`dxdiag` reports static capacity, not live utilization. The question
of whether DXGI exposes dynamic memory (free right now, as a real-time
signal) was not resolved in this investigation. The relevant API is
`IDXGIAdapter3::QueryVideoMemoryInfo`; Microsoft documents that it
returns current memory usage and budget, but empirical confirmation
requires a small program. That verification is deferred until the
first integration attempt.

If `QueryVideoMemoryInfo` returns dynamic data, DXGI becomes a drop-in
replacement for `nvidia-smi` on Windows, with the bonus of covering
both vendors in a single call.

If it only returns capacity, DXGI still has value for adapter
enumeration, but per-GPU live telemetry would need vendor-specific
fallbacks (nvidia-smi for NVIDIA, Intel-specific API for Intel iGPU).

## Recommendation for v22 implementation order

When APX v22 (multi-backend foundation) is attempted:

1. **First probe:** DXGI on Windows. No install, covers both vendors,
   confirms the abstraction design works.
2. **If DXGI gives dynamic memory:** promote it to the primary probe
   on Windows, keep nvidia-smi as a cross-platform fallback for Linux
   and as a second opinion.
3. **If DXGI is capacity-only:** keep nvidia-smi for NVIDIA live data,
   add Intel-specific fallback (OpenCL or Level Zero) for Intel live
   data.
4. **Last resort:** OpenCL or Level Zero, only if the above combination
   leaves gaps that matter.

## Non-goals for this investigation

- No code was written. This is research, not integration.
- No APIs were installed beyond what ships with the OS.
- Apple Metal and AMD ROCm were explicitly out of scope; they do not
  apply to the host used for investigation.

## Open items for future work

- Confirm empirically that `IDXGIAdapter3::QueryVideoMemoryInfo`
  returns dynamic memory data (small Rust POC using the `windows`
  crate, to be written when v22 starts).
- Test OpenCL after installing Intel OpenCL Runtime, to compare
  overhead and information quality against DXGI.
- Define the abstraction trait (`MemoryProbe` or similar) that
  vendor-specific implementations will satisfy.
