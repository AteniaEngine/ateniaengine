//! Disk-backed tensor tier (M3-e.11.1).
//!
//! Infrastructure for the third storage tier introduced in M3-e.11:
//! on-disk spillover for tensors when both VRAM and RAM are saturated
//! (the "dual-pressure" scenario where `Degrade` alone cannot free
//! enough memory).
//!
//! **This sub-milestone is standalone.** The module is self-contained
//! and its types are not yet plumbed through `TensorStorage`, the
//! reaction loop, or the `Tensor` API. Integration lands in M3-e.11.2
//! when `TensorStorage::Disk(DiskTensorHandle)` is added and the match
//! arms across the codebase are updated. The current commit only
//! exercises the infrastructure in its own unit tests.
//!
//! ## File format — raw bytes, dtype on the handle (M4.7.4.a)
//!
//! Tensors are serialized as the raw little-endian (native) byte
//! layout of their underlying buffer:
//!
//! - [`DiskDtype::F32`]: 4 bytes per element, the original M3-e.11.1
//!   format. Bit-exact backward-compatible.
//! - [`DiskDtype::BF16`]: 2 bytes per element. The buffer is the
//!   `Vec<u16>` storage shared with [`crate::tensor::TensorStorage::CpuBf16`].
//!
//! No header, no magic number, no dtype tag inside the file. The
//! dtype lives on the [`DiskTensorHandle`] alongside `numel`, the
//! same pattern that already worked in M3-e.11.1. Rationale:
//!
//! - **Zero new deps**: the project does not use `serde`, `bincode`,
//!   `safetensors`, or `bytemuck`; adding one for ephemeral spillover
//!   files would be disproportionate.
//! - **Files are ephemeral**: written by one process, read by the
//!   same process (or cleaned up by the next one via [`gc_orphan_disk_tensors`]).
//!   No cross-process, cross-architecture or cross-version
//!   compatibility requirement.
//! - **I/O speed**: the disk tier is the slowest tier by definition.
//!   Every byte of format overhead translates to latency at the
//!   worst possible moment. Raw bytes give the best I/O throughput
//!   achievable without OS tuning.
//!
//! The tensor's shape / layout live on the owning [`Tensor`][crate::tensor::Tensor]
//! and are not duplicated in the file. The [`DiskTensorHandle`] caches
//! `numel` and `dtype` for a size-check at read time; the shape is
//! never persisted.
//!
//! ## Cache directory
//!
//! See [`default_cache_dir`] for the resolution order. Notably, the
//! default on Linux is **not** `/tmp` / `std::env::temp_dir()` because
//! on most modern distros `/tmp` is mounted as `tmpfs` — RAM-backed
//! storage. Spilling VRAM→RAM→tmpfs is spilling to RAM under a
//! different name, which defeats the purpose of the disk tier. Users
//! with a RAM-backed `$XDG_CACHE_HOME` are similarly affected; this
//! is not detected automatically and should be documented to
//! operators.
//!
//! ## File lifecycle — Arc + Drop
//!
//! [`DiskTensorHandle`] wraps an [`Arc<InnerDiskFile>`]. Cloning a
//! handle is cheap (atomic refcount) and shares the underlying file;
//! the last clone's drop removes the file via best-effort
//! `std::fs::remove_file` (errors are swallowed — an on-start GC via
//! [`gc_orphan_disk_tensors`] is the safety net). The pattern mirrors
//! `Arc<InnerGpuPtr>` in `gpu::tensor` (M3-d.1).
//!
//! A hard crash of the process (SIGKILL, power loss, panic with
//! `panic = "abort"`) bypasses Drop and leaves orphan files in the
//! cache directory. [`gc_orphan_disk_tensors`] cleans those up at
//! start; the wire-up to call it automatically lands in a later
//! sub-milestone (likely e.11.5 as part of the reactive-context
//! initialization).

use std::fs;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use uuid::Uuid;

/// M4.7.4.b — chunk size used by the streaming readers
/// (`read_f32_tensor`, `read_bf16_tensor`). 4 MiB is large enough
/// to amortise per-syscall overhead on Windows / NTFS and to land
/// inside one CPU L3 slice on the dev-target hardware, but small
/// enough that the per-call peak allocation stays bounded
/// regardless of tensor size — a 13B-class checkpoint that
/// previously required a single 26 GB `Vec<u8>` at restore time
/// now fits inside one of these chunks plus the destination
/// `Vec<f32>` / `Vec<u16>`.
const STREAM_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Read `expected_bytes` from `path` directly into the byte view
/// of `out`, in [`STREAM_CHUNK_BYTES`]-sized batches (M4.7.4.b).
///
/// `out` must already be sized so its byte-view length equals
/// `expected_bytes`; the caller is responsible for the
/// `numel * width` arithmetic and for picking a `T` whose layout
/// matches the on-disk encoding.
///
/// Why a streaming read instead of a single `fs::read`:
///
/// - **Bounded peak allocation.** Whole-file `fs::read` allocates
///   a fresh `Vec<u8>` of the file's full size before producing
///   the destination `Vec<T>`, so the restore of a 26 GB tensor
///   takes ~52 GB of transient RAM at the worst possible moment
///   (the moment the reactive loop fired the spill because RAM
///   was already saturated). Streaming reads write directly into
///   the destination buffer, capping transient overhead at one
///   chunk plus the `File` handle.
/// - **Open-and-stream is idiomatic in `std`.** `File::open` +
///   `read_exact` works on every platform without a new dep
///   (`memmap2` was rejected — see module docstring rationale).
/// - **NTFS / USB-tier friendliness.** A 4 MiB chunk is two NTFS
///   default extents; sequential reads at this granularity
///   saturate consumer NVMe and stay well within the SLC-cache
///   working set on USB-attached storage.
///
/// On a partial / truncated file the final `read_exact` returns
/// `UnexpectedEof`; the caller surfaces this as `InvalidData` with
/// the size-mismatch message it already produced before M4.7.4.b
/// (the size check is now a redundant safety net inside the
/// `_into` helper itself).
fn stream_read_exact_into_bytes(
    path: &Path,
    expected_bytes: usize,
    out_bytes: &mut [u8],
) -> io::Result<()> {
    if out_bytes.len() != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "stream_read_exact_into_bytes: out buffer is {} bytes, expected {}",
                out_bytes.len(),
                expected_bytes
            ),
        ));
    }

    // Validate file size up-front so a truncated file is rejected
    // before we issue the first read syscall — preserves the
    // pre-M4.7.4.b error semantics (size mismatch returns
    // `InvalidData`, not `UnexpectedEof`).
    let metadata = fs::metadata(path)?;
    let actual_bytes = metadata.len();
    if actual_bytes != expected_bytes as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "disk tensor size mismatch: file={} bytes, expected={} bytes",
                actual_bytes, expected_bytes
            ),
        ));
    }

    let mut file = fs::File::open(path)?;

    let mut offset = 0usize;
    while offset < expected_bytes {
        let chunk_end = (offset + STREAM_CHUNK_BYTES).min(expected_bytes);
        let chunk = &mut out_bytes[offset..chunk_end];
        file.read_exact(chunk)?;
        offset = chunk_end;
    }
    Ok(())
}

/// **NUMERIC-POLICY-2** — write a per-row int8-quantised `[rows, cols]` weight
/// to a **deterministic** `path` as `[rows × f32 scales][numel × i8 quants]`,
/// returning a **persistent** [`DiskDtype::QInt8`] handle. Halves the bf16 tier
/// (~1 byte/element), so the resolve reads — and the antivirus scans — far fewer
/// bytes (the MOE-IO-1 bottleneck). Lossy; certified by tolerance vs Certified.
pub fn write_qint8_tensor_named(
    path: &Path,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> io::Result<DiskTensorHandle> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let (q, scales) = quantize_per_row_i8(data, rows, cols);
    let mut buf = Vec::with_capacity(rows * 4 + q.len());
    for s in &scales {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf.extend(q.iter().map(|&v| v as u8));
    fs::write(path, &buf)?;
    Ok(DiskTensorHandle {
        inner: Arc::new(InnerDiskFile {
            path: path.to_path_buf(),
            numel: rows * cols,
            dtype: DiskDtype::QInt8,
            persistent: true,
        }),
    })
}

/// **NUMERIC-POLICY-2** — read a per-row int8 tier file (written by
/// [`write_qint8_tensor_named`]) and **dequantize to f32**. `rows` is the quant
/// axis (the tensor's first dim); `cols = numel / rows`. Validates the on-disk
/// size (`rows*4 + numel`); a mismatch → `InvalidData` (caller falls back).
pub fn read_qint8_to_f32(handle: &DiskTensorHandle, rows: usize) -> io::Result<Vec<f32>> {
    let numel = handle.numel();
    if rows == 0 || numel % rows != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "qint8: bad rows for numel"));
    }
    let cols = numel / rows;
    let expected = qint8_disk_bytes(numel, rows);
    let bytes = fs::read(handle.path())?;
    if bytes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("qint8 file {:?}: {} bytes, expected {expected}", handle.path(), bytes.len()),
        ));
    }
    let mut scales = vec![0.0f32; rows];
    for (r, s) in scales.iter_mut().enumerate() {
        let b = r * 4;
        *s = f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]);
    }
    let q: Vec<i8> = bytes[rows * 4..].iter().map(|&b| b as i8).collect();
    Ok(dequantize_per_row_i8(&q, &scales, rows, cols))
}

/// **NUMERIC-POLICY-2** — per-row **symmetric int8** quantization of a row-major
/// `[rows, cols]` weight: each row gets one f32 `scale = max(|row|)/127`, and
/// `q = round(w/scale)` clamped to `[-127, 127]`. Returns `(q, scales)`. An
/// all-zero row uses `scale = 1` (`q = 0`). The quant axis is the **output row**
/// (the standard, most accurate per-output-channel choice for a weight matmul).
pub fn quantize_per_row_i8(data: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![1.0f32; rows];
    for r in 0..rows {
        let base = r * cols;
        let amax = data[base..base + cols].iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        scales[r] = scale;
        let inv = 1.0 / scale;
        for c in 0..cols {
            q[base + c] = (data[base + c] * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

/// **NUMERIC-POLICY-2** — dequantize a per-row symmetric int8 weight back to f32:
/// `w[r,c] = q[r,c] * scales[r]`. Inverse of [`quantize_per_row_i8`].
pub fn dequantize_per_row_i8(q: &[i8], scales: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let s = scales[r];
        for c in 0..cols {
            out[base + c] = q[base + c] as f32 * s;
        }
    }
    out
}

/// On-disk dtype of a spilled tensor (M4.7.4.a).
///
/// Lives on the [`DiskTensorHandle`], not in the file itself. The
/// reader picks the right deserializer (`read_f32_tensor` or
/// `read_bf16_tensor`) based on the value cached here at write time.
///
/// Adding a new variant is a backward-compatible operation: existing
/// `F32` handles keep working, the on-disk byte layout for the new
/// variant is the raw little-endian buffer of the corresponding Rust
/// integer width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiskDtype {
    /// 4-byte IEEE-754 single-precision float, the original M3-e.11.1
    /// format. Round-trip via [`write_f32_tensor`] / [`read_f32_tensor`].
    F32,
    /// 2-byte bfloat16, raw `u16` bits; the same encoding the
    /// [`crate::tensor::TensorStorage::CpuBf16`] variant carries.
    /// Round-trip via [`write_bf16_tensor`] / [`read_bf16_tensor`].
    BF16,
    /// **NUMERIC-POLICY-2** — per-row symmetric int8 (lossy). On disk:
    /// `[rows × f32 scales][rows*cols × i8 quants]` = `rows*4 + numel` bytes
    /// (≈ 1 byte/element + a tiny scale header). Round-trip via
    /// [`write_qint8_tensor_named`] / [`read_qint8_to_f32`]; needs `rows` (the
    /// quant axis) to split the header, supplied by the caller from the shape.
    QInt8,
}

impl DiskDtype {
    /// Bytes per **element** for the dense dtypes (F32 = 4, BF16 = 2). `QInt8`
    /// is **not** a fixed per-element width (it carries a per-row scale header),
    /// so it returns the 1-byte quant width; use [`qint8_disk_bytes`] for the
    /// full on-disk size of a quantised tensor.
    pub fn bytes_per_element(self) -> usize {
        match self {
            DiskDtype::F32 => 4,
            DiskDtype::BF16 => 2,
            DiskDtype::QInt8 => 1,
        }
    }
}

/// **NUMERIC-POLICY-2** — full on-disk byte size of a per-row int8 tensor:
/// `rows` f32 scales + `numel` i8 quants.
pub fn qint8_disk_bytes(numel: usize, rows: usize) -> usize {
    rows * 4 + numel
}

/// Owning record of an on-disk tensor file. Drop removes the file
/// on a best-effort basis. Not cloneable on its own — use
/// [`DiskTensorHandle`] (which wraps it in `Arc`) if you need
/// shared ownership.
#[derive(Debug)]
pub struct InnerDiskFile {
    path: PathBuf,
    numel: usize,
    /// On-disk dtype. Defaults to [`DiskDtype::F32`] for handles
    /// constructed via the legacy `write_f32_tensor` path, preserving
    /// bit-exact M3-e.11.1 behaviour.
    dtype: DiskDtype,
    /// **MOE-PROD-4** — when `true` the file is **persistent**: its `Drop`
    /// does **not** delete it, so a deterministically-named tier file survives
    /// process exit and can be reused by a later run. Ephemeral (UUID) handles
    /// keep `false` and are still auto-deleted on drop (unchanged behaviour).
    persistent: bool,
}

impl Drop for InnerDiskFile {
    fn drop(&mut self) {
        // Persistent tier files (MOE-PROD-4) are owned by the on-disk tier
        // manifest, not by the handle lifetime — never auto-delete them.
        if self.persistent {
            return;
        }
        // Best-effort delete. We intentionally ignore the result:
        // - `NotFound` happens if the file was already gc'd by a
        //   concurrent process or manually deleted by the user.
        // - `PermissionDenied` indicates the fs policy changed
        //   mid-process; the on-start GC will pick up the file
        //   eventually.
        // - Any other error (disk unplugged, fs corruption) we
        //   cannot reasonably recover from here; aborting the
        //   program because a temp file refused to be deleted
        //   would be worse than the leak.
        let _ = fs::remove_file(&self.path);
    }
}

/// Shared handle to an on-disk tensor. Cheap to clone (atomic
/// refcount); the underlying file is removed when the last clone
/// drops.
#[derive(Clone, Debug)]
pub struct DiskTensorHandle {
    inner: Arc<InnerDiskFile>,
}

impl DiskTensorHandle {
    /// Path to the backing file. Stable for the lifetime of the
    /// handle (and its clones).
    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    /// Number of elements stored in the file. Cached at creation
    /// time; used by [`read_f32_tensor`] / [`read_bf16_tensor`] for
    /// a size consistency check (`numel * dtype.bytes_per_element()
    /// == file_size`).
    pub fn numel(&self) -> usize {
        self.inner.numel
    }

    /// On-disk dtype of this handle (M4.7.4.a). Determines whether
    /// callers should route to [`read_f32_tensor`] or
    /// [`read_bf16_tensor`]. Set at write time; immutable for the
    /// lifetime of the file.
    pub fn dtype(&self) -> DiskDtype {
        self.inner.dtype
    }
}

/// Resolve the cache directory where disk-tier files live.
///
/// Resolution order:
/// 1. `ATENIA_DISK_TIER_DIR` environment variable, if set.
/// 2. **Windows**: `%LOCALAPPDATA%\Atenia\cache\`.
/// 3. **Linux / Unix**: `$XDG_CACHE_HOME/atenia/` or
///    `$HOME/.cache/atenia/`. **Not** `/tmp` — on most Linux
///    distros `/tmp` is `tmpfs` (RAM-backed) and would defeat
///    the point of the disk tier.
/// 4. **macOS**: `$HOME/Library/Caches/atenia/` (via the same
///    `$HOME` branch used on other Unix platforms).
/// 5. **Last-resort fallback**: `std::env::temp_dir().join("atenia")`.
///    Always succeeds but may be RAM-backed on some configurations.
///
/// This function does not create the directory — callers that
/// write files via [`write_f32_tensor`] rely on that function's
/// own `create_dir_all` to handle creation.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(v) = std::env::var("ATENIA_DISK_TIER_DIR") {
        return PathBuf::from(v);
    }
    #[cfg(windows)]
    {
        if let Ok(la) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(la).join("Atenia").join("cache");
        }
    }
    #[cfg(unix)]
    {
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(xdg).join("atenia");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".cache").join("atenia");
        }
    }
    // Last-resort fallback. Works on every platform but offers
    // no guarantees about disk-backing.
    std::env::temp_dir().join("atenia")
}

/// Serialize an `f32` slice to a fresh file under `cache_dir` and
/// return a [`DiskTensorHandle`] owning the result.
///
/// The file name is `tensor_<uuid>.bin`, unique across threads and
/// processes. The cache directory is created (recursively) if it
/// does not already exist.
///
/// On success the handle's Drop owns the lifecycle — the file is
/// removed when the last clone of the handle goes out of scope.
/// On failure (write error, permission denied, disk full) no file
/// is left behind (Rust's `fs::write` does not create partial
/// files reliably across platforms, but any partial file is a
/// candidate for the next GC sweep anyway).
pub fn write_f32_tensor(cache_dir: &Path, data: &[f32]) -> io::Result<DiskTensorHandle> {
    fs::create_dir_all(cache_dir)?;

    let name = format!("tensor_{}.bin", Uuid::new_v4());
    let path = cache_dir.join(name);

    // SAFETY: `f32` is `Copy + 'static` with a stable memory layout
    // (4 bytes, IEEE 754). Re-interpreting `&[f32]` as `&[u8]` of
    // four times the length is the documented standard way to read
    // its bytes; `slice::from_raw_parts` is safe as long as
    // (a) the pointer and length are well-formed, which `data.as_ptr()`
    //     and `data.len() * 4` guarantee,
    // (b) the memory is valid for reads for the full length, which
    //     holds because we are inside the lifetime of `data`,
    // (c) the underlying allocation is not mutated while the slice
    //     lives, which our signature enforces via `&[f32]`.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    fs::write(&path, bytes)?;

    Ok(DiskTensorHandle {
        inner: Arc::new(InnerDiskFile {
            path,
            numel: data.len(),
            dtype: DiskDtype::F32,
            persistent: false,
        }),
    })
}

/// **MOE-PROD-4** — write an f32 tensor to a **deterministic** `path` (not a
/// UUID) and return a **persistent** handle (its `Drop` does not delete the
/// file). Used by the persistent MoE tier so a later run can reuse the file.
/// Creates the parent directory if needed.
pub fn write_f32_tensor_named(path: &Path, data: &[f32]) -> io::Result<DiskTensorHandle> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // SAFETY: identical justification to `write_f32_tensor` — `&[f32]` → `&[u8]`
    // byte view over the same allocation for the duration of the borrow.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    fs::write(path, bytes)?;
    Ok(DiskTensorHandle {
        inner: Arc::new(InnerDiskFile {
            path: path.to_path_buf(),
            numel: data.len(),
            dtype: DiskDtype::F32,
            persistent: true,
        }),
    })
}

/// **MOE-PROD-5** — read a persistent f32 tier file at `path` back into a
/// `Vec<f32>`, validating that its byte length matches `expected_numel`
/// elements. Errors (missing / wrong size / unreadable) let the caller fall
/// back to the shard path — never a silent wrong answer.
pub fn read_f32_named(path: &Path, expected_numel: usize) -> io::Result<Vec<f32>> {
    // **MOE-PROD-7** — stream directly into the destination `Vec<f32>` byte view
    // (4 MiB chunks) instead of `fs::read` + a per-element `from_le_bytes` loop.
    // Bit-identical on little-endian targets (the on-disk bytes are the native
    // LE layout); bounded peak allocation; ~the bandwidth of a memcpy.
    let expected_bytes = expected_numel.checked_mul(4).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "tier numel * 4 overflows usize")
    })?;
    let mut out = vec![0f32; expected_numel];
    // SAFETY: `out` owns `expected_numel` f32 = `expected_bytes` bytes,
    // contiguous; the byte view aliases the same allocation for the read only.
    let out_bytes = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, std::mem::size_of_val(&*out))
    };
    stream_read_exact_into_bytes(path, expected_bytes, out_bytes)?;
    Ok(out)
}

/// **MOE-PROD-7** — read a persistent tier file to `Vec<f32>`, detecting the
/// on-disk dtype from the file's byte length: `numel*4` → F32 (direct),
/// `numel*2` → BF16 (upcast to f32 via the SIMD `bf16_decode_bulk`, the same
/// lossless `<<16` expansion the `ensure_cpu` disk arm uses). Any other length
/// is corruption → `InvalidData` (caller falls back to the certified path).
/// Used for the warm-reconstruction backend tensors so they too can be bf16.
pub fn read_named_to_f32(path: &Path, numel: usize) -> io::Result<Vec<f32>> {
    let len = fs::metadata(path)?.len();
    if len == (numel as u64) * 4 {
        read_f32_named(path, numel)
    } else if len == (numel as u64) * 2 {
        let mut bits = vec![0u16; numel];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(bits.as_mut_ptr() as *mut u8, std::mem::size_of_val(&*bits))
        };
        stream_read_exact_into_bytes(path, numel * 2, bytes)?;
        let mut out = vec![0f32; numel];
        crate::simd_kernels::avx2::bf16_decode_bulk(&bits, &mut out);
        Ok(out)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tier file {path:?}: {len} bytes, expected {} (f32) or {} (bf16)", numel * 4, numel * 2),
        ))
    }
}

/// **MOE-PROD-4** — wrap an **existing** persistent f32 tier file as a handle
/// **without** writing it (the reuse fast-path: the deterministically-named
/// file already holds the right bytes). `numel` is the element count the caller
/// expects; the file is assumed valid (the caller validates presence + size).
pub fn open_existing_f32(path: &Path, numel: usize) -> DiskTensorHandle {
    open_existing(path, numel, DiskDtype::F32)
}

/// **MOE-PROD-6** — generalised [`open_existing_f32`]: wrap an existing
/// persistent tier file as a handle of the given on-disk `dtype`
/// (F32 = 4 B/elem, BF16 = 2 B/elem) **without** writing it. The
/// caller validates presence + byte length (`numel * dtype.bytes_per_element()`)
/// before relying on the handle; materialising it later (`ensure_cpu`)
/// dispatches on the dtype and upcasts BF16 → F32 losslessly.
pub fn open_existing(path: &Path, numel: usize, dtype: DiskDtype) -> DiskTensorHandle {
    DiskTensorHandle {
        inner: Arc::new(InnerDiskFile {
            path: path.to_path_buf(),
            numel,
            dtype,
            persistent: true,
        }),
    }
}

/// **MOE-PROD-6** — write a BF16 tensor (`&[u16]` raw bf16 bits) to a
/// **deterministic** `path` and return a **persistent** handle (its `Drop`
/// does not delete the file). The on-disk size is `numel * 2` — half the f32
/// tier. Mirrors [`write_f32_tensor_named`] for the bf16 expert tier; a later
/// run reuses the file via [`open_existing`] with [`DiskDtype::BF16`].
pub fn write_bf16_tensor_named(path: &Path, data: &[u16]) -> io::Result<DiskTensorHandle> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // SAFETY: `u16` is `Copy + 'static`, 2 bytes, stable layout — same byte-view
    // justification as `write_f32_tensor_named` with width 2.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    fs::write(path, bytes)?;
    Ok(DiskTensorHandle {
        inner: Arc::new(InnerDiskFile {
            path: path.to_path_buf(),
            numel: data.len(),
            dtype: DiskDtype::BF16,
            persistent: true,
        }),
    })
}

/// Serialize a `bf16`-as-`u16` slice to a fresh file under
/// `cache_dir` and return a [`DiskTensorHandle`] tagged with
/// [`DiskDtype::BF16`] (M4.7.4.a).
///
/// Mirrors [`write_f32_tensor`]: same UUID file naming, same
/// `create_dir_all` behaviour, same "the file is owned by the
/// returned handle's Arc Drop". The only difference is that each
/// element is serialized as 2 raw little-endian bytes instead of 4.
///
/// The 2-byte width matches [`crate::tensor::TensorStorage::CpuBf16`]
/// so the spill / restore cycle preserves the M4.7.2 50% RAM
/// footprint win on disk too — a 13B-class checkpoint stays at
/// ~26 GB on disk under BF16 spill instead of inflating to ~52 GB
/// via F32 upcast.
pub fn write_bf16_tensor(cache_dir: &Path, data: &[u16]) -> io::Result<DiskTensorHandle> {
    fs::create_dir_all(cache_dir)?;

    let name = format!("tensor_{}.bin", Uuid::new_v4());
    let path = cache_dir.join(name);

    // SAFETY: `u16` is `Copy + 'static`, 2 bytes wide with a stable
    // memory layout. The justification is identical to the f32 path
    // above with width 2 instead of 4. Lifetime / aliasing constraints
    // are enforced by the `&[u16]` argument type.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    fs::write(&path, bytes)?;

    Ok(DiskTensorHandle {
        inner: Arc::new(InnerDiskFile {
            path,
            numel: data.len(),
            dtype: DiskDtype::BF16,
            persistent: false,
        }),
    })
}

/// **M7.1** — zero-host-copy variant of [`write_bf16_tensor`].
/// Takes a `&[u8]` of raw BF16 bytes (typically a slice into a
/// safetensors reader's owned buffer) plus the logical element
/// count, and writes the bytes directly to disk without first
/// materialising a host-side `Vec<u16>` or `Vec<f32>`.
///
/// This is the load-time path used by
/// `WeightMapper::load_into_with_residency_plan` when:
/// - the source dtype is BF16,
/// - the parameter has no `LoadTransform`s registered, and
/// - the planner assigned it to [`crate::gpu::tier_plan::Tier::Disk`].
///
/// Under those conditions, the entire weight write happens
/// without ever materialising a host-side F32 transient or a
/// secondary `Vec<u16>`. The peak host-RAM cost is the
/// safetensors reader's owned byte buffer (which the caller
/// already paid for), no more — closing risk R3 (BSOD-class
/// peak-RAM regression) for the Disk-tier loader path.
///
/// # Errors
///
/// - `InvalidInput` if `raw_bytes.len() != numel * 2`.
/// - Whatever `fs::create_dir_all` / `fs::write` surface
///   (write error, permission denied, disk full).
///
/// # Bit-exactness
///
/// The on-disk byte layout is identical to what
/// [`write_bf16_tensor`] would produce given the same logical
/// values, because both functions just memcpy the BF16 bits to
/// disk in little-endian `u16` order. A round-trip via
/// [`read_bf16_tensor`] returns the same `Vec<u16>` either way.
pub fn write_bf16_from_raw_bytes(
    cache_dir: &Path,
    raw_bytes: &[u8],
    numel: usize,
) -> io::Result<DiskTensorHandle> {
    if raw_bytes.len() != numel * 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "write_bf16_from_raw_bytes: raw_bytes.len()={} but \
                 numel={} requires {} bytes",
                raw_bytes.len(),
                numel,
                numel * 2
            ),
        ));
    }

    fs::create_dir_all(cache_dir)?;

    let name = format!("tensor_{}.bin", Uuid::new_v4());
    let path = cache_dir.join(name);
    fs::write(&path, raw_bytes)?;

    Ok(DiskTensorHandle {
        inner: Arc::new(InnerDiskFile {
            path,
            numel,
            dtype: DiskDtype::BF16,
            persistent: false,
        }),
    })
}

/// Deserialize the file referenced by `handle` into an owned
/// `Vec<f32>`. Validates that the on-disk byte count matches
/// `handle.numel() * 4` and returns `InvalidData` on mismatch —
/// typical cause is a file truncated by a crash mid-write.
///
/// Asserts `handle.dtype() == DiskDtype::F32`. Use
/// [`read_bf16_tensor`] for BF16-tagged handles. The dtype check
/// is a defence-in-depth against a caller routing a BF16 handle
/// here by mistake — the size validation alone would silently
/// "succeed" on a file whose `numel * 4` happens to match a
/// different real size, returning garbage floats.
pub fn read_f32_tensor(handle: &DiskTensorHandle) -> io::Result<Vec<f32>> {
    if handle.dtype() != DiskDtype::F32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "read_f32_tensor called on {:?} handle; route via read_bf16_tensor instead",
                handle.dtype()
            ),
        ));
    }
    let expected_bytes = handle.numel().checked_mul(4).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "disk tensor numel * 4 overflows usize",
        )
    })?;

    let mut out = vec![0f32; handle.numel()];
    // SAFETY: `out` has capacity `handle.numel() * 4` bytes; the
    // byte-view length matches `expected_bytes`. `out` and the file
    // do not alias by definition. `f32` does not require alignment
    // beyond 4 bytes; the byte-view re-interpretation imposes no
    // additional constraint.
    let out_bytes = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, std::mem::size_of_val(&*out))
    };
    stream_read_exact_into_bytes(handle.path(), expected_bytes, out_bytes)?;
    Ok(out)
}

/// Deserialize the file referenced by `handle` into an owned
/// `Vec<u16>` of bf16 bits (M4.7.4.a).
///
/// Mirrors [`read_f32_tensor`] with width 2: validates
/// `bytes.len() == handle.numel() * 2`, then `copy_nonoverlapping`
/// into a fresh `Vec<u16>`. Asserts `handle.dtype() ==
/// DiskDtype::BF16`. No upcast to f32 here — that is the caller's
/// (and `ensure_cpu`'s) responsibility, and intentionally so:
/// callers that just need to keep the BF16 view around (e.g. moving
/// an in-flight BF16 parameter back from disk into a `CpuBf16`
/// storage in a future M5 milestone) avoid the doubled allocation.
pub fn read_bf16_tensor(handle: &DiskTensorHandle) -> io::Result<Vec<u16>> {
    if handle.dtype() != DiskDtype::BF16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "read_bf16_tensor called on {:?} handle; route via read_f32_tensor instead",
                handle.dtype()
            ),
        ));
    }
    let expected_bytes = handle.numel().checked_mul(2).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "disk tensor numel * 2 overflows usize",
        )
    })?;

    let mut out = vec![0u16; handle.numel()];
    // SAFETY: same justification as `read_f32_tensor`'s byte-view
    // re-interpretation, with width 2 instead of 4. `u16` has
    // alignment 2; the byte-view imposes no additional constraint.
    let out_bytes = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, std::mem::size_of_val(&*out))
    };
    stream_read_exact_into_bytes(handle.path(), expected_bytes, out_bytes)?;
    Ok(out)
}

/// **M8.7.0** — zero-host-allocation streaming reader of a BF16
/// disk tensor into a caller-supplied byte buffer.
///
/// Mirrors [`read_bf16_tensor`] but skips the owned `Vec<u16>`
/// allocation. Used by the Disk → GPU JIT pipeline
/// (`cuda_matmul_disk_streamed_bf16`) where the destination is a
/// VRAM staging slot and the host buffer is a per-call transient
/// recycled across matmuls; allocating a fresh `Vec<u16>` for
/// every matmul (one per Disk-tier weight on the 13B forward —
/// 197 calls per token) would defeat the M4.7.4.b bounded-peak
/// streaming property the rest of `disk_tier` is built around.
///
/// # Contract
///
/// - `handle.dtype() == DiskDtype::BF16` (otherwise `InvalidData`).
/// - `out_bytes.len() == handle.numel() * 2`.
/// - On success the entire buffer is filled in 4 MiB chunks via
///   the same [`stream_read_exact_into_bytes`] path used by
///   [`read_bf16_tensor`] / [`read_f32_tensor`].
///
/// On any I/O / size error the buffer contents are unspecified
/// (the caller should treat the dispatch as failed and fall back).
pub fn read_bf16_raw_bytes(handle: &DiskTensorHandle, out_bytes: &mut [u8]) -> io::Result<()> {
    if handle.dtype() != DiskDtype::BF16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "read_bf16_raw_bytes called on {:?} handle; \
                 only BF16 disk tensors are supported on this path",
                handle.dtype()
            ),
        ));
    }
    let expected_bytes = handle.numel().checked_mul(2).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "disk tensor numel * 2 overflows usize",
        )
    })?;
    stream_read_exact_into_bytes(handle.path(), expected_bytes, out_bytes)
}

/// Remove orphan `tensor_*.bin` files from `cache_dir` that are
/// older than `max_age_minutes`. Returns the count removed.
///
/// Orphans arise when a previous process crashed without running
/// [`InnerDiskFile::Drop`] (SIGKILL, power loss, abort-on-panic).
/// "Older than N minutes" is the heuristic for "belongs to a
/// terminated process": a live process touches its files within
/// seconds of starting, so a threshold of 10 minutes comfortably
/// avoids deleting files owned by a concurrent sibling process
/// that just started.
///
/// The function only inspects entries whose file name starts with
/// `tensor_` and ends with `.bin`. Other files in the directory
/// are left alone — users or other tools may share the cache
/// directory.
///
/// Errors from [`fs::remove_file`] are swallowed per-entry
/// (best-effort GC, same policy as [`InnerDiskFile::Drop`]);
/// errors from [`fs::read_dir`] and reading `metadata` propagate.
pub fn gc_orphan_disk_tensors(cache_dir: &Path, max_age_minutes: u64) -> io::Result<usize> {
    gc_orphan_disk_tensors_older_than(
        cache_dir,
        Duration::from_secs(max_age_minutes.saturating_mul(60)),
    )
}

/// Implementation-level variant of [`gc_orphan_disk_tensors`] that
/// accepts a `Duration` directly. Exposed `pub(crate)` so tests can
/// drive the GC with sub-minute thresholds without sleeping for
/// minutes.
pub(crate) fn gc_orphan_disk_tensors_older_than(
    cache_dir: &Path,
    max_age: Duration,
) -> io::Result<usize> {
    let entries = match fs::read_dir(cache_dir) {
        Ok(e) => e,
        // If the cache dir does not exist, there is nothing to
        // clean — not an error.
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };

    let now = SystemTime::now();
    let mut removed = 0usize;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue, // skip unreadable entries
        };
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !name.starts_with("tensor_") || !name.ends_with(".bin") {
            continue;
        }

        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let modified = match metadata.modified() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let age = match now.duration_since(modified) {
            Ok(d) => d,
            // File's modification time is in the future (clock
            // skew). Leave it alone; a benign file will age
            // normally on subsequent sweeps.
            Err(_) => continue,
        };

        if age > max_age {
            // Best-effort. A failure here (permissions, file locked
            // on Windows) is logged by being absent from the count,
            // not propagated — the next sweep will retry.
            if fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
    }

    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::thread;

    /// Serialize env-var manipulation across tests in this file.
    /// Two tests below set/clear `ATENIA_DISK_TIER_DIR`, and that is
    /// process-global state; parallel tests without a lock race on it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Build a throwaway cache directory unique to this test run.
    /// Cleanup is best-effort (`remove_dir_all` at the end); the
    /// GC and/or OS will sweep leftovers in a pinch.
    fn test_cache_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("atenia_test_{}_{}", label, Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create test cache dir");
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let dir = test_cache_dir("roundtrip");
        let data: Vec<f32> = vec![1.0, -2.5, 3.14159, 0.0, f32::INFINITY, -0.0];
        let handle = write_f32_tensor(&dir, &data).expect("write must succeed");
        assert_eq!(handle.numel(), data.len());
        assert!(handle.path().exists(), "file must exist after write");

        let read = read_f32_tensor(&handle).expect("read must succeed");
        // f32 equality is bit-level; -0.0 and 0.0 compare equal, but
        // we assert the whole vector as-is which is what matters.
        assert_eq!(read.len(), data.len());
        for (i, (a, b)) in data.iter().zip(read.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "mismatch at index {}", i);
        }

        drop(handle);
        cleanup(&dir);
    }

    #[test]
    fn test_read_numel_mismatch() {
        let dir = test_cache_dir("numel_mismatch");
        let data: Vec<f32> = vec![1.0; 10];
        let good = write_f32_tensor(&dir, &data).expect("write ok");

        // Build a synthetic handle that claims a different numel —
        // pointing to the same file. The read must refuse.
        let bad = DiskTensorHandle {
            inner: Arc::new(InnerDiskFile {
                path: good.path().to_path_buf(),
                numel: 7, // file has 10 floats, not 7
                dtype: DiskDtype::F32,
                persistent: false,
            }),
        };
        let result = read_f32_tensor(&bad);
        assert!(result.is_err(), "read with wrong numel must fail");
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        // Drop `bad` first so the file is still present for `good`'s
        // Drop to handle (InnerDiskFile::Drop is best-effort — even
        // if bad deleted it, good would just find no-such-file).
        drop(bad);
        drop(good);
        cleanup(&dir);
    }

    #[test]
    fn test_handle_drop_deletes_file() {
        let dir = test_cache_dir("drop_deletes");
        let data: Vec<f32> = vec![42.0; 4];
        let path_holder = {
            let handle = write_f32_tensor(&dir, &data).expect("write ok");
            let p = handle.path().to_path_buf();
            assert!(p.exists(), "file must exist while handle is alive");
            p
            // handle drops here
        };
        assert!(
            !path_holder.exists(),
            "file must be gone after handle drop, but still present at {:?}",
            path_holder
        );
        cleanup(&dir);
    }

    #[test]
    fn test_handle_clone_shares_file() {
        let dir = test_cache_dir("clone_shares");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let h1 = write_f32_tensor(&dir, &data).expect("write ok");
        let path = h1.path().to_path_buf();
        let h2 = h1.clone();

        // Both handles point to the same file.
        assert_eq!(h1.path(), h2.path());
        assert!(path.exists());

        // Drop one — file still alive.
        drop(h1);
        assert!(
            path.exists(),
            "file must survive while h2 still holds a ref"
        );

        // Drop the last — file gone.
        drop(h2);
        assert!(
            !path.exists(),
            "file must be removed after last clone drops"
        );

        cleanup(&dir);
    }

    #[test]
    fn test_default_cache_dir_honors_env_var() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        let sentinel = std::env::temp_dir().join("atenia_test_env_sentinel");
        let sentinel_str = sentinel.to_string_lossy().into_owned();
        // SAFETY (2024 edition): env::set_var/remove_var are unsafe
        // because they cross thread boundaries without
        // synchronization; we hold `ENV_LOCK` for the duration and
        // restore state before releasing it.
        unsafe {
            std::env::set_var("ATENIA_DISK_TIER_DIR", &sentinel_str);
        }

        let got = default_cache_dir();
        assert_eq!(got, sentinel);

        unsafe {
            std::env::remove_var("ATENIA_DISK_TIER_DIR");
        }
    }

    #[test]
    fn test_default_cache_dir_fallback() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        // Ensure the override is NOT set.
        unsafe {
            std::env::remove_var("ATENIA_DISK_TIER_DIR");
        }

        let got = default_cache_dir();
        let as_str = got.to_string_lossy();

        // Platform-dependent expectation: we assert that the path
        // contains "atenia" (as dir component) and lies under a
        // plausible root. Too-strict string matching is fragile
        // across Windows vs Unix path styles.
        assert!(
            as_str.to_lowercase().contains("atenia"),
            "expected path to contain 'atenia', got {}",
            as_str
        );
        #[cfg(windows)]
        {
            // On Windows the expected root is %LOCALAPPDATA% unless
            // missing; either way the tail is "Atenia\cache" or
            // ends in "atenia".
            assert!(
                as_str.contains("Atenia") || as_str.contains("atenia"),
                "expected Windows default to mention Atenia, got {}",
                as_str
            );
        }
        #[cfg(unix)]
        {
            // On Unix the default should be under $HOME/.cache or
            // $XDG_CACHE_HOME or the temp fallback. All of them
            // end with "atenia" as the last component.
            assert_eq!(
                got.file_name().and_then(|s| s.to_str()),
                Some("atenia"),
                "expected Unix default to end in 'atenia', got {}",
                as_str
            );
        }
    }

    #[test]
    fn test_gc_removes_old_files() {
        let dir = test_cache_dir("gc_old");
        let handle = write_f32_tensor(&dir, &[1.0, 2.0]).expect("write ok");
        let path = handle.path().to_path_buf();
        // Detach the path ownership from the handle so the file is
        // NOT removed via Drop before we get to GC it. Forget the
        // Arc via mem::forget.
        std::mem::forget(handle);

        // Wait a little so the file has a non-zero age on any
        // filesystem (some fs round mtime to the nearest second).
        thread::sleep(Duration::from_millis(50));

        // Threshold just under the sleep: anything older than 10ms
        // is "old enough". Our file is 50ms+ old, gets swept.
        let removed = gc_orphan_disk_tensors_older_than(&dir, Duration::from_millis(10))
            .expect("gc must succeed");
        assert_eq!(removed, 1, "gc should have removed the stale file");
        assert!(!path.exists(), "file should be gone after gc");

        cleanup(&dir);
    }

    #[test]
    fn test_gc_preserves_new_files() {
        let dir = test_cache_dir("gc_new");
        let handle = write_f32_tensor(&dir, &[3.0, 4.0]).expect("write ok");
        let path = handle.path().to_path_buf();

        // Threshold of 10 minutes — our just-written file is way
        // younger than that. GC must leave it alone.
        let removed = gc_orphan_disk_tensors(&dir, 10).expect("gc must succeed");
        assert_eq!(removed, 0, "gc must not touch fresh files");
        assert!(path.exists(), "file must still exist");

        drop(handle);
        cleanup(&dir);
    }

    #[test]
    fn test_gc_ignores_non_tensor_files() {
        let dir = test_cache_dir("gc_ignore");

        // Drop an unrelated file in the same dir.
        let stranger = dir.join("unrelated.txt");
        fs::write(&stranger, b"hello").expect("write stranger");

        // Wait for it to age, then GC with tiny threshold.
        thread::sleep(Duration::from_millis(50));
        let removed = gc_orphan_disk_tensors_older_than(&dir, Duration::from_millis(10))
            .expect("gc must succeed");

        assert_eq!(removed, 0, "gc must ignore files that are not tensor_*.bin");
        assert!(
            stranger.exists(),
            "stranger file must survive gc, path {:?} gone",
            stranger
        );

        cleanup(&dir);
    }

    #[test]
    fn test_gc_on_missing_directory_is_ok() {
        // If the cache dir does not exist yet, GC is a no-op.
        let dir = std::env::temp_dir().join(format!("atenia_test_missing_{}", Uuid::new_v4()));
        assert!(!dir.exists(), "pre-condition: dir must not exist");
        let removed = gc_orphan_disk_tensors(&dir, 10).expect("gc must return Ok on missing dir");
        assert_eq!(removed, 0);
    }

    // ------------------------------------------------------------
    // M4.7.4.a — BF16 round-trip + dtype-tag tests
    // ------------------------------------------------------------

    #[test]
    fn test_bf16_write_read_roundtrip_bit_exact() {
        let dir = test_cache_dir("bf16_roundtrip");
        // Mix of representative bf16 patterns: zero, +/- finite,
        // a denormal-ish small magnitude, +/- infinity, NaN. The
        // encoding is the upper 16 bits of the f32 representation,
        // produced inline so we don't depend on a runtime helper.
        let f32_pattern: [f32; 6] = [
            0.0_f32,
            -2.5_f32,
            3.14159_f32,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7FC00000), // canonical quiet NaN
        ];
        let bf16: Vec<u16> = f32_pattern
            .iter()
            .map(|f| (f.to_bits() >> 16) as u16)
            .collect();

        let handle = write_bf16_tensor(&dir, &bf16).expect("write must succeed");
        assert_eq!(handle.numel(), bf16.len());
        assert_eq!(handle.dtype(), DiskDtype::BF16);
        assert!(handle.path().exists(), "file must exist after write");

        let read = read_bf16_tensor(&handle).expect("read must succeed");
        assert_eq!(read.len(), bf16.len());
        // Bit-exact: bf16 is a u16 lane, no rounding involved on
        // round-trip.
        assert_eq!(read, bf16);

        drop(handle);
        cleanup(&dir);
    }

    #[test]
    fn test_bf16_file_is_half_the_size_of_f32() {
        // 50 % footprint saving is the M4.7.2 contract carried into
        // the disk tier. Any future format change must keep this.
        let dir = test_cache_dir("bf16_size");
        let n = 1024_usize;
        let f32_data = vec![0.5_f32; n];
        let bf16_data: Vec<u16> = f32_data
            .iter()
            .map(|f| (f.to_bits() >> 16) as u16)
            .collect();

        let h_f32 = write_f32_tensor(&dir, &f32_data).expect("f32 write");
        let h_bf16 = write_bf16_tensor(&dir, &bf16_data).expect("bf16 write");

        let f32_size = fs::metadata(h_f32.path()).expect("f32 meta").len();
        let bf16_size = fs::metadata(h_bf16.path()).expect("bf16 meta").len();

        assert_eq!(f32_size as usize, n * 4, "f32 file = numel * 4 bytes");
        assert_eq!(bf16_size as usize, n * 2, "bf16 file = numel * 2 bytes");
        assert_eq!(
            bf16_size * 2,
            f32_size,
            "bf16 file must be exactly half the f32 file"
        );

        drop(h_f32);
        drop(h_bf16);
        cleanup(&dir);
    }

    #[test]
    fn test_read_f32_refuses_bf16_handle() {
        // Cross-routing must be a hard error, not a silent garbage
        // read. This is the dtype defence-in-depth from the
        // function docstring.
        let dir = test_cache_dir("cross_route_bf16");
        let bf16 = vec![0x4040_u16; 4]; // arbitrary
        let handle = write_bf16_tensor(&dir, &bf16).expect("write ok");
        let result = read_f32_tensor(&handle);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        drop(handle);
        cleanup(&dir);
    }

    #[test]
    fn test_read_bf16_refuses_f32_handle() {
        // Symmetric counterpart of the f32 → bf16 cross-route guard.
        let dir = test_cache_dir("cross_route_f32");
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let handle = write_f32_tensor(&dir, &data).expect("write ok");
        let result = read_bf16_tensor(&handle);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        drop(handle);
        cleanup(&dir);
    }

    #[test]
    fn test_bf16_handle_drop_deletes_file() {
        // Same Arc-Drop lifecycle as the f32 path; verifies the
        // BF16 write path does not regress the cleanup contract.
        let dir = test_cache_dir("bf16_drop");
        let path_holder = {
            let handle = write_bf16_tensor(&dir, &[0xBF80_u16; 8]).expect("write ok");
            let p = handle.path().to_path_buf();
            assert!(p.exists());
            p
        };
        assert!(
            !path_holder.exists(),
            "bf16 file must be removed after handle drop"
        );
        cleanup(&dir);
    }

    #[test]
    fn test_bf16_size_mismatch_detected() {
        // Synthesize a BF16 handle that claims more elements than
        // the file actually holds. Read must surface InvalidData.
        let dir = test_cache_dir("bf16_size_mismatch");
        let bf16 = vec![0x4080_u16; 6]; // file has 6 u16 = 12 bytes
        let good = write_bf16_tensor(&dir, &bf16).expect("write ok");

        let bad = DiskTensorHandle {
            inner: Arc::new(InnerDiskFile {
                path: good.path().to_path_buf(),
                numel: 9, // file has 6 u16, not 9
                dtype: DiskDtype::BF16,
                persistent: false,
            }),
        };
        let result = read_bf16_tensor(&bad);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);

        drop(bad);
        drop(good);
        cleanup(&dir);
    }

    #[test]
    fn test_per_row_i8_quant_dequant_roundtrip_bounded() {
        // NUMERIC-POLICY-2: per-row symmetric int8 quant error is <= scale/2 =
        // (amax/127)/2 per element. Verify bit patterns dequantise within that.
        let rows = 4usize;
        let cols = 5usize;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.37 - 3.1).collect();
        let (q, scales) = quantize_per_row_i8(&data, rows, cols);
        assert_eq!(q.len(), rows * cols);
        assert_eq!(scales.len(), rows);
        let dq = dequantize_per_row_i8(&q, &scales, rows, cols);
        for r in 0..rows {
            let amax = data[r * cols..r * cols + cols].iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let tol = (amax / 127.0) / 2.0 + 1e-6;
            for c in 0..cols {
                let d = (dq[r * cols + c] - data[r * cols + c]).abs();
                assert!(d <= tol, "row {r} col {c}: |{d}| > {tol}");
            }
        }
        // An all-zero row → scale 1, q 0, exact.
        let z = vec![0.0f32; cols];
        let (qz, sz) = quantize_per_row_i8(&z, 1, cols);
        assert!(qz.iter().all(|&v| v == 0));
        assert_eq!(sz[0], 1.0);
        assert_eq!(dequantize_per_row_i8(&qz, &sz, 1, cols), z);
    }

    #[test]
    fn test_read_named_to_f32_detects_dtype_by_size() {
        // MOE-PROD-7: a dtype-detecting named reader for the warm backend.
        let dir = test_cache_dir("named_to_f32");
        // f32 file: read back exactly.
        let f32_data: Vec<f32> = vec![1.5, -2.0, 0.0, 3.25];
        let f32_path = dir.join("a_f32.bin");
        write_f32_tensor_named(&f32_path, &f32_data).unwrap();
        let got_f32 = read_named_to_f32(&f32_path, f32_data.len()).unwrap();
        assert_eq!(got_f32, f32_data);

        // bf16 file (bf16-source values): upcast back to the exact f32.
        let src: Vec<f32> = f32_data
            .iter()
            .map(|f| f32::from_bits(f.to_bits() & 0xFFFF_0000))
            .collect();
        let bits: Vec<u16> = src.iter().map(|f| (f.to_bits() >> 16) as u16).collect();
        let bf16_path = dir.join("a_bf16.bin");
        write_bf16_tensor_named(&bf16_path, &bits).unwrap();
        let got_bf16 = read_named_to_f32(&bf16_path, src.len()).unwrap();
        assert_eq!(got_bf16, src, "bf16 file must upcast to the exact f32 values");
        // bf16 file is half the size of the f32 file.
        assert_eq!(fs::metadata(&bf16_path).unwrap().len() * 2, fs::metadata(&f32_path).unwrap().len());

        // A size matching neither dtype is rejected.
        assert!(read_named_to_f32(&f32_path, f32_data.len() + 1).is_err());

        cleanup(&dir);
    }

    #[test]
    fn test_disk_dtype_bytes_per_element() {
        assert_eq!(DiskDtype::F32.bytes_per_element(), 4);
        assert_eq!(DiskDtype::BF16.bytes_per_element(), 2);
    }

    // ------------------------------------------------------------
    // M4.7.4.b — streaming reader tests
    // ------------------------------------------------------------

    #[test]
    fn test_streaming_read_handles_tensor_larger_than_chunk() {
        // The streaming reader's contract is "produces the same
        // Vec<f32> regardless of how many chunks the file is split
        // into". Build a tensor whose byte size strictly exceeds
        // STREAM_CHUNK_BYTES so the read loop has to iterate more
        // than once, then verify bit-exact bytes.
        let dir = test_cache_dir("stream_large");

        // 1.5x chunk size in bytes → 1.5 MiB worth of f32 elements.
        // (STREAM_CHUNK_BYTES is currently 4 MiB; this gives us a
        // 1.5-iteration-equivalent read at f32 width.)
        let n_floats = (STREAM_CHUNK_BYTES / 4) + (STREAM_CHUNK_BYTES / 8);
        let data: Vec<f32> = (0..n_floats).map(|i| (i as f32) * 0.001 - 7.5).collect();

        let handle = write_f32_tensor(&dir, &data).expect("write");
        let read = read_f32_tensor(&handle).expect("streaming read");

        assert_eq!(read.len(), data.len());
        // Bit-exact f32 round-trip — the bytes on disk are a memcpy
        // of `data`, the streaming reader is also a memcpy, so the
        // result is identical.
        for (i, (a, b)) in data.iter().zip(read.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "streaming read mismatch at index {}",
                i
            );
        }

        drop(handle);
        cleanup(&dir);
    }

    #[test]
    fn test_streaming_bf16_read_handles_tensor_larger_than_chunk() {
        // Counterpart for bf16: 1.5x chunk size at u16 width.
        let dir = test_cache_dir("stream_large_bf16");
        let n_u16 = (STREAM_CHUNK_BYTES / 2) + (STREAM_CHUNK_BYTES / 4);
        // Pseudo-deterministic pattern so a bug that swapped two
        // chunks would surface as a value mismatch instead of an
        // off-by-one.
        let data: Vec<u16> = (0..n_u16).map(|i| (i as u16).wrapping_mul(73)).collect();

        let handle = write_bf16_tensor(&dir, &data).expect("write");
        let read = read_bf16_tensor(&handle).expect("streaming read");

        assert_eq!(read.len(), data.len());
        assert_eq!(read, data, "bf16 streaming round-trip must be bit-exact");

        drop(handle);
        cleanup(&dir);
    }

    #[test]
    fn test_streaming_read_truncated_file_surfaces_invalid_data() {
        // Truncate the file *after* the handle was created so the
        // size mismatch is real on the filesystem but the handle's
        // cached numel still claims the original count. The reader
        // must surface InvalidData (not UnexpectedEof) — preserves
        // the pre-streaming error semantics.
        let dir = test_cache_dir("stream_truncated");
        let data: Vec<f32> = vec![1.0_f32; 1024];
        let handle = write_f32_tensor(&dir, &data).expect("write");

        // Truncate to half the original byte size.
        let path = handle.path().to_path_buf();
        let f = fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for truncate");
        f.set_len((data.len() * 4 / 2) as u64).expect("truncate");
        drop(f);

        let result = read_f32_tensor(&handle);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidData,
            "truncated read must surface InvalidData, got: {:?}",
            err
        );

        drop(handle);
        cleanup(&dir);
    }

    #[test]
    fn test_f32_handle_default_dtype_is_f32() {
        // Backward-compatibility lock: the legacy `write_f32_tensor`
        // path must produce a handle tagged DiskDtype::F32 so any
        // code that reaches for `handle.dtype()` (M4.7.4.d and on)
        // dispatches correctly without an explicit migration.
        let dir = test_cache_dir("default_dtype");
        let h = write_f32_tensor(&dir, &[1.0_f32, 2.0, 3.0]).expect("write");
        assert_eq!(h.dtype(), DiskDtype::F32);
        drop(h);
        cleanup(&dir);
    }
}
