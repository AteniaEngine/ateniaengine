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
//! ## File format — raw f32 bytes
//!
//! Tensors are serialized as the raw little-endian (native) byte
//! layout of their `Vec<f32>` contents. No header, no magic number,
//! no dtype tag. Rationale:
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
//! The tensor's shape / dtype / layout live on the owning [`Tensor`][crate::tensor::Tensor]
//! and are not duplicated in the file. The [`DiskTensorHandle`] caches
//! `numel` for a size-check at read time; the shape is never persisted.
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
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use uuid::Uuid;

/// Owning record of an on-disk tensor file. Drop removes the file
/// on a best-effort basis. Not cloneable on its own — use
/// [`DiskTensorHandle`] (which wraps it in `Arc`) if you need
/// shared ownership.
#[derive(Debug)]
pub struct InnerDiskFile {
    path: PathBuf,
    numel: usize,
}

impl Drop for InnerDiskFile {
    fn drop(&mut self) {
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

    /// Number of `f32` elements stored in the file. Cached at
    /// creation time; used by [`read_f32_tensor`] for a size
    /// consistency check.
    pub fn numel(&self) -> usize {
        self.inner.numel
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
pub fn write_f32_tensor(
    cache_dir: &Path,
    data: &[f32],
) -> io::Result<DiskTensorHandle> {
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
        }),
    })
}

/// Deserialize the file referenced by `handle` into an owned
/// `Vec<f32>`. Validates that the on-disk byte count matches
/// `handle.numel() * 4` and returns `InvalidData` on mismatch —
/// typical cause is a file truncated by a crash mid-write.
pub fn read_f32_tensor(handle: &DiskTensorHandle) -> io::Result<Vec<f32>> {
    let bytes = fs::read(handle.path())?;
    let expected_bytes = handle.numel().checked_mul(4).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "disk tensor numel * 4 overflows usize",
        )
    })?;
    if bytes.len() != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "disk tensor size mismatch: file={} bytes, expected={} bytes ({} floats)",
                bytes.len(),
                expected_bytes,
                handle.numel()
            ),
        ));
    }

    let mut out = vec![0f32; handle.numel()];
    // SAFETY: `out` has capacity `handle.numel() * 4` bytes, and
    // we have just checked that `bytes.len()` equals that. `bytes`
    // and `out` do not alias (distinct allocations). `f32` has no
    // alignment constraints beyond 4 bytes; `copy_nonoverlapping`
    // does not require source alignment, only that the ranges are
    // valid for reads and writes respectively, which they are.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            bytes.len(),
        );
    }
    Ok(out)
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
pub fn gc_orphan_disk_tensors(
    cache_dir: &Path,
    max_age_minutes: u64,
) -> io::Result<usize> {
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
        let dir = std::env::temp_dir()
            .join(format!("atenia_test_{}_{}", label, Uuid::new_v4()));
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
        let removed = gc_orphan_disk_tensors_older_than(
            &dir,
            Duration::from_millis(10),
        )
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
        let removed =
            gc_orphan_disk_tensors(&dir, 10).expect("gc must succeed");
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
        let removed = gc_orphan_disk_tensors_older_than(
            &dir,
            Duration::from_millis(10),
        )
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
        let dir = std::env::temp_dir()
            .join(format!("atenia_test_missing_{}", Uuid::new_v4()));
        assert!(!dir.exists(), "pre-condition: dir must not exist");
        let removed =
            gc_orphan_disk_tensors(&dir, 10).expect("gc must return Ok on missing dir");
        assert_eq!(removed, 0);
    }
}
