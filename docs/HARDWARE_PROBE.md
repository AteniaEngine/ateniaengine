# Hardware Probe

Cross-vendor GPU enumeration binary for Atenia. Reports every GPU
visible to the host and augments NVIDIA entries with compute
capability, VRAM, and CUDA runtime version. Intended as the first
step toward Atenia's multi-vendor support (APX v22+).

## What it does

1. Enumerates every graphics adapter visible to **wgpu** (Vulkan on
   Linux, DX12 on Windows, Metal on macOS, with GL/Vulkan fallbacks).
   Covers NVIDIA, AMD, Intel, and Apple GPUs in one call.
2. Augments NVIDIA entries via **NVML** when the NVIDIA driver is
   present: compute capability (e.g. `8.9` for Ada Lovelace), total
   and free VRAM in megabytes, driver version, CUDA runtime version.
3. Collects system info: OS name and version, arch, hostname, total
   RAM.
4. Emits either a human-readable text report (default) or a JSON
   document (`--output json`).

What it does **not** do:

- Run compute benchmarks. Kernel execution belongs in a separate
  binary that can assume a compute runtime is installed; the probe
  must work on hosts that don't have CUDA / ROCm / oneAPI yet.
- Detect every runtime. Only NVIDIA augmentation is wired today.
  AMD (ROCm), Apple (Metal compute), and Intel (Level Zero / oneAPI)
  augmentations are deferred — wgpu alone already gives us
  vendor / model / backend for them.
- Classify into Atenia support tiers. The `atenia_support_tier`
  field is always `null` in this version; once
  `hardware_compatibility.toml` exists, a follow-up change will fill
  it in from a vendor/device lookup.

## Build

The probe lives behind a Cargo feature flag (`hw-probe`) so normal
library builds do not pull in wgpu's heavy dependency tree.

```
cargo build --release --bin hardware_probe --features hw-probe
```

First build compiles wgpu + nvml-wrapper + os_info + serde, which
takes ~60-90 seconds. Subsequent builds reuse the cached artifacts
and finish in a few seconds.

## Run

```
./target/release/hardware_probe                 # text output
./target/release/hardware_probe --output json   # JSON output
./target/release/hardware_probe --help          # usage
```

## Sample output (text)

```
Atenia hardware probe v0.1.0
Probed at: 2026-04-24T15:42:00Z UTC

System:
  OS:       windows (Windows 11 (24H2))
  Arch:     x86_64
  Hostname: atenia-dev-notebook
  RAM:      32768 MB

GPUs detected: 2

GPU 0:
  Vendor:   NVIDIA (0x10DE)
  Device:   0x2786
  Name:     NVIDIA GeForce RTX 4070 Laptop GPU
  Type:     discrete
  Backend:  dx12
  Driver:   NVIDIA
  Version:  560.70
  VRAM:     8188 MB (7512 MB free)
  Compute:  8.9 (NVIDIA CC)
  Runtime:  CUDA 13.2
  Atenia:   tier classification not available (hardware_compatibility.toml pending)

GPU 1:
  Vendor:   Intel (0x8086)
  Device:   0xA7A1
  Name:     Intel(R) UHD Graphics
  Type:     integrated
  Backend:  dx12
  Driver:   Intel
  Version:  31.0.101.5382
  Atenia:   tier classification not available (hardware_compatibility.toml pending)
```

## Sample output (JSON, abbreviated)

```json
{
  "probe_version": "0.1.0",
  "probed_at_unix_secs": 1761320520,
  "probed_at_iso8601": "2026-04-24T15:42:00Z",
  "system": {
    "os": "windows",
    "os_version": "Windows 11 (24H2)",
    "arch": "x86_64",
    "hostname": "atenia-dev-notebook",
    "ram_total_mb": 32768
  },
  "gpus": [
    {
      "index": 0,
      "vendor": "NVIDIA",
      "vendor_id": 4318,
      "device_id": 10118,
      "name": "NVIDIA GeForce RTX 4070 Laptop GPU",
      "device_type": "discrete",
      "backend": "dx12",
      "driver": "NVIDIA",
      "driver_version": "560.70",
      "vram_mb_total": 8188,
      "vram_mb_free": 7512,
      "runtime_detected": ["CUDA 13.2"],
      "compute_capability": "8.9",
      "atenia_support_tier": null
    }
  ],
  "warnings": []
}
```

## Known limitations

- **Headless Linux servers**: if the host has a GPU but no graphics
  driver (Vulkan / OpenGL absent, only a compute-only CUDA driver),
  wgpu's `enumerate_adapters` returns 0 and the probe reports no
  GPUs. A warning surfaces in the `warnings` array. NVML would still
  detect NVIDIA in that case; a future revision could add synthetic
  GPU entries built from NVML alone when wgpu misses them.
- **Multi-GPU with identical models**: two identical NVIDIA cards
  share vendor/device IDs. The NVML matching code attaches one NVML
  device per wgpu adapter in enumeration order. This is correct for
  the usual notebook/desktop case; on multi-GPU servers with mixed
  driver ordering the mapping can drift. Not a target today.
- **macOS Apple Silicon + Asahi Linux**: wgpu via Metal on macOS is
  tested upstream; wgpu via Vulkan on Asahi Linux (Apple Silicon
  with the Asahi GPU driver) is not a target and has not been
  validated.
- **Shared memory semantics**: on Windows hosts with an iGPU, DXGI
  reports a large "shared memory" pool that is the same physical RAM
  both GPUs and the CPU already share. The probe reports only the
  dedicated VRAM for integrated GPUs (or leaves it `None` where wgpu
  does not expose it). See `docs/RESEARCH_INTEL_APIS.md` for the
  full semantic discussion.

## Contributing hardware reports

The probe is most useful if we accumulate reports from a variety of
hosts. If you run the binary on uncommon hardware (workstations with
multiple AMD cards, Apple Silicon M3 Ultra, Intel Arc discrete, etc.)
and want to help tune Atenia's future `hardware_compatibility.toml`,
attach the JSON output to an issue. Redact hostname / RAM if those
are sensitive.
