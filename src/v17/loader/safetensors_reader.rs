//! Safetensors reader (M4-a).
//!
//! First sub-phase of the M4 ModelLoader milestone: a standalone
//! reader for the HuggingFace safetensors format. Parses the header,
//! exposes an owned handle over the file bytes, and offers both
//! iterator and by-name access to tensor entries.
//!
//! # What this module does NOT do (yet)
//!
//! - Integrate with the Atenia graph (M4-b / M4-c).
//! - Convert BF16 or F16 tensors to `Vec<f32>` — those dtypes surface
//!   as [`LoaderError::UnsupportedDType`] for now. M4-d extends
//!   conversion support.
//! - Memory-map the file. For M4-a the full file is read into a
//!   `Vec<u8>`. `memmap2` integration is deferred until a target
//!   model shows the copy-into-RAM cost in the profile.
//!
//! # Design
//!
//! `SafetensorsReader` is an **owned** handle: it takes ownership of
//! the file bytes (or reads them from a path) and parses the header
//! once at construction. The `safetensors` crate's `SafeTensors<'data>`
//! view has a lifetime tied to the input bytes; we use it only
//! transiently during construction to validate the format and extract
//! shape / dtype / offset metadata into owned `EntryOffsets`
//! structures. After construction the crate's view is dropped, and
//! subsequent queries borrow from `self.bytes` directly.
//!
//! This avoids self-referential lifetimes in the public API and lets
//! callers pass the reader around without carrying the raw bytes
//! buffer as a separate argument.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::tensor::DType;

use super::loader_errors::LoaderError;

/// Offsets and metadata of a single tensor inside the safetensors
/// body. Stored owned so the reader does not depend on the lifetime
/// of the `safetensors::SafeTensors` deserialization view.
#[derive(Debug, Clone)]
struct EntryOffsets {
    shape: Vec<usize>,
    dtype: DType,
    /// Offset where this tensor's payload begins — into
    /// `SafetensorsReader::bytes` when `from_decoded == false`, or into
    /// `SafetensorsReader::decoded` when `from_decoded == true`.
    body_start: usize,
    /// Offset (exclusive) where this tensor's payload ends.
    body_end: usize,
    /// **FP8-SAFETENSORS-1** — `true` when this entry was an on-disk FP8
    /// tensor decoded to F32 at construction; its `dtype` is then `F32`
    /// and its bytes live in the side `decoded` buffer, so everything
    /// downstream sees a plain F32 tensor (no FP8 in the graph/kernels).
    from_decoded: bool,
}

/// **STREAMING-LOADER-1** — how the reader holds the file bytes. `open`
/// memory-maps the file (file-backed, reclaimable pages → low peak committed
/// RAM); `from_bytes` keeps an owned heap buffer (for `.bin`-transcoded /
/// network / FP8-decoded inputs). Both deref to `&[u8]`, so the rest of the
/// reader is backing-agnostic.
enum Backing {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Backing::Owned(v) => v,
            Backing::Mapped(m) => m,
        }
    }
}

impl std::fmt::Debug for Backing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backing::Owned(v) => write!(f, "Owned({} bytes)", v.len()),
            Backing::Mapped(m) => write!(f, "Mapped({} bytes)", m.len()),
        }
    }
}

/// Owned safetensors reader. Construct via [`Self::open`] (memory-mapped) or
/// [`Self::from_bytes`] (owned); query via [`Self::iter`], [`Self::get`], and
/// [`Self::metadata`].
#[derive(Debug)]
pub struct SafetensorsReader {
    bytes: Backing,
    /// **FP8-SAFETENSORS-1** — side buffer holding F32-decoded bytes for
    /// tensors that were FP8 on disk. Empty when the file has no FP8
    /// tensors (zero overhead for the common case).
    decoded: Vec<u8>,
    entries: HashMap<String, EntryOffsets>,
    /// Tensor names in declaration order (the order they appear in
    /// the header JSON). `HashMap` iteration is non-deterministic, so
    /// we preserve declaration order separately — useful for tests
    /// and for loaders that want to stream weights in a predictable
    /// sequence.
    names_in_order: Vec<String>,
    /// Parsed `__metadata__` field from the header, if present.
    metadata: Option<HashMap<String, String>>,
}

/// Borrowed view of a single tensor entry. Lifetime is tied to the
/// reader, so [`Self::raw_bytes`] remains valid for as long as the
/// reader is alive.
#[derive(Debug)]
pub struct TensorEntry<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub dtype: DType,
    pub raw_bytes: &'a [u8],
}

impl SafetensorsReader {
    /// Open a file from disk and parse its safetensors header. The
    /// entire file is read into memory (no mmap in M4-a).
    pub fn open(path: &Path) -> Result<Self, LoaderError> {
        if !path.exists() {
            return Err(LoaderError::FileNotFound(
                path.to_string_lossy().to_string(),
            ));
        }
        // **STREAMING-LOADER-1** — memory-map by default (low peak RAM). The
        // explicit `ATENIA_DISABLE_MMAP=1` opt-out and the automatic fallback
        // on mmap failure both go through the byte-identical owned read path,
        // so behaviour is never silently different — only the RAM profile is.
        if std::env::var("ATENIA_DISABLE_MMAP").as_deref() == Ok("1") {
            let bytes = fs::read(path).map_err(|e| LoaderError::IoError(e.to_string()))?;
            return Self::from_backing(Backing::Owned(bytes));
        }
        let file = std::fs::File::open(path).map_err(|e| LoaderError::IoError(e.to_string()))?;
        // SAFETY: read-only mapping of a model file for the duration of load.
        // The file is not mutated by Atenia; external truncation while mapped
        // is the standard documented mmap caveat (same assumption the
        // `safetensors` crate's own mmap API makes).
        match unsafe { memmap2::Mmap::map(&file) } {
            Ok(mmap) => Self::from_backing(Backing::Mapped(mmap)),
            Err(e) => {
                eprintln!(
                    "[ATENIA] mmap failed for {} ({e}); falling back to full read \
                     (higher peak RAM, identical bytes).",
                    path.display()
                );
                let bytes = fs::read(path).map_err(|e| LoaderError::IoError(e.to_string()))?;
                Self::from_backing(Backing::Owned(bytes))
            }
        }
    }

    /// Construct a reader from an in-memory byte buffer. Same
    /// validation as [`Self::open`] but skips the filesystem step —
    /// useful for tests and for callers that obtain bytes over the
    /// network.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LoaderError> {
        Self::from_backing(Backing::Owned(bytes))
    }

    /// Backing-agnostic construction. Parses the header + tensor offsets from
    /// `backing` (owned `Vec` or memory-map), then takes ownership of it.
    fn from_backing(backing: Backing) -> Result<Self, LoaderError> {
        let bytes: &[u8] = &backing;
        // Parse the header once to extract the optional
        // `__metadata__` map — the `SafeTensors<'data>` view produced
        // by `deserialize` does not expose a `metadata()` accessor in
        // crate version 0.4, so we call `read_metadata` separately.
        // The extra pass over the first 8 + header_len bytes is
        // negligible; the body itself is not re-scanned.
        let (_hdr_len, hdr_metadata) =
            safetensors::SafeTensors::read_metadata(bytes).map_err(|e| {
                LoaderError::InvalidFormat(format!("safetensors read_metadata: {:?}", e))
            })?;
        let metadata: Option<HashMap<String, String>> = hdr_metadata.metadata().clone();

        // `safetensors::SafeTensors::deserialize` validates the
        // header-size prefix, parses the JSON, and checks that every
        // declared tensor's `data_offsets` falls within the body.
        let view = safetensors::SafeTensors::deserialize(bytes)
            .map_err(|e| LoaderError::InvalidFormat(format!("safetensors deserialize: {:?}", e)))?;

        // Compute the header-size prefix so we can translate
        // body-relative offsets (what `TensorView::data()` returns a
        // slice over) to absolute offsets into `bytes`. The prefix is
        // the first 8 bytes of the file (u64 little-endian header
        // length) + the header itself.
        if bytes.len() < 8 {
            return Err(LoaderError::InvalidFormat(
                "file shorter than 8-byte header-size prefix".to_string(),
            ));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
        let body_base = 8usize.checked_add(header_len).ok_or_else(|| {
            LoaderError::InvalidFormat(format!(
                "header_len ({}) overflows usize when added to 8-byte prefix",
                header_len
            ))
        })?;
        if body_base > bytes.len() {
            return Err(LoaderError::InvalidFormat(format!(
                "declared header length ({}) + 8-byte prefix exceeds file size ({})",
                header_len,
                bytes.len()
            )));
        }

        let mut entries: HashMap<String, EntryOffsets> = HashMap::new();
        let mut names_in_order: Vec<String> = Vec::new();
        let mut decoded: Vec<u8> = Vec::new();

        for (name, tv) in view.tensors() {
            let shape: Vec<usize> = tv.shape().to_vec();
            let data = tv.data();

            // **FP8-SAFETENSORS-1** — decode FP8 (E4M3 / E5M2) to F32 at
            // construction into the side `decoded` buffer, so the entry
            // is presented as a plain F32 tensor. The graph, kernels,
            // adapters and tier planner never see an FP8 dtype.
            use safetensors::Dtype as St;
            let st_dtype = tv.dtype();
            if matches!(st_dtype, St::F8_E4M3 | St::F8_E5M2) {
                let elems: usize = shape.iter().product();
                if data.len() != elems {
                    return Err(LoaderError::InvalidFormat(format!(
                        "tensor '{}': FP8 body carries {} bytes but shape {:?} implies {} \
                         elements (FP8 is 1 byte/element)",
                        name,
                        data.len(),
                        shape,
                        elems
                    )));
                }
                let body_start = decoded.len();
                decoded.reserve(elems * 4);
                for &b in data {
                    let f = match st_dtype {
                        St::F8_E4M3 => fp8_e4m3_to_f32(b),
                        _ => fp8_e5m2_to_f32(b),
                    };
                    decoded.extend_from_slice(&f.to_le_bytes());
                }
                entries.insert(
                    name.to_string(),
                    EntryOffsets {
                        shape,
                        dtype: DType::F32,
                        body_start,
                        body_end: decoded.len(),
                        from_decoded: true,
                    },
                );
                names_in_order.push(name.to_string());
                continue;
            }

            let dtype = map_dtype(st_dtype)
                .map_err(|msg| LoaderError::InvalidFormat(format!("tensor '{}': {}", name, msg)))?;

            // `data` is a slice of `bytes`; compute its absolute
            // offset range by pointer arithmetic. This is safe
            // because `SafeTensors::deserialize` guarantees that
            // `data` is a subslice of the buffer we passed in.
            let data_start = data.as_ptr() as usize - bytes.as_ptr() as usize;
            let body_start = data_start;
            let body_end = body_start + data.len();

            if body_end > bytes.len() {
                return Err(LoaderError::InvalidFormat(format!(
                    "tensor '{}': body_end {} exceeds buffer length {}",
                    name,
                    body_end,
                    bytes.len()
                )));
            }

            entries.insert(
                name.to_string(),
                EntryOffsets {
                    shape,
                    dtype,
                    body_start,
                    body_end,
                    from_decoded: false,
                },
            );
            names_in_order.push(name.to_string());
        }

        // `view` is dropped here — from now on we read from `bytes`
        // (or `decoded`) directly via the offsets we captured.
        drop(view);

        Ok(Self {
            bytes: backing,
            decoded,
            entries,
            names_in_order,
            metadata,
        })
    }

    /// Number of tensors in the file.
    pub fn len(&self) -> usize {
        self.names_in_order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names_in_order.is_empty()
    }

    /// Iterate tensor entries in declaration order (the order they
    /// appear in the header JSON).
    pub fn iter(&self) -> impl Iterator<Item = TensorEntry<'_>> {
        self.names_in_order.iter().map(move |name| {
            let entry = &self.entries[name];
            let buf: &[u8] = if entry.from_decoded { &self.decoded } else { &self.bytes };
            TensorEntry {
                name: name.as_str(),
                shape: entry.shape.as_slice(),
                dtype: entry.dtype,
                raw_bytes: &buf[entry.body_start..entry.body_end],
            }
        })
    }

    /// Look up a tensor by name. Returns `None` if the tensor is not
    /// present. The returned `TensorEntry`'s fields borrow from the
    /// reader, so `name` has the reader's lifetime (not the caller's
    /// input string) — the name is resolved against the owned
    /// `names_in_order` table.
    pub fn get(&self, name: &str) -> Option<TensorEntry<'_>> {
        let owned_name = self.names_in_order.iter().find(|n| n.as_str() == name)?;
        let entry = &self.entries[owned_name];
        let buf: &[u8] = if entry.from_decoded { &self.decoded } else { &self.bytes };
        Some(TensorEntry {
            name: owned_name.as_str(),
            shape: entry.shape.as_slice(),
            dtype: entry.dtype,
            raw_bytes: &buf[entry.body_start..entry.body_end],
        })
    }

    /// Return the parsed `__metadata__` field if the file contained
    /// one. The safetensors format reserves `__metadata__` in the
    /// header JSON for free-form key/value strings the producer wants
    /// to ship alongside the weights (tokenizer identifier, training
    /// recipe, checkpoint hash, etc.).
    pub fn metadata(&self) -> Option<&HashMap<String, String>> {
        self.metadata.as_ref()
    }
}

impl TensorEntry<'_> {
    /// Decode this entry's raw bytes into a `Vec<f32>`.
    ///
    /// Supported dtypes:
    /// - `DType::F32`: pass-through decode via `f32::from_le_bytes`.
    /// - `DType::BF16` (M4-d): host-side upcast via `f32::from_bits((bf16 as u32) << 16)`.
    ///   BF16 is IEEE 754 single-precision truncated to the top 16 bits
    ///   (1 sign + 8 exponent + 7 mantissa), so the conversion is a pure
    ///   zero-extension of the low 16 bits — exact for every BF16 value,
    ///   no rounding.
    /// - `DType::F16` (M4-d): host-side conversion via the `half` crate's
    ///   `f16::from_bits(...).to_f32()`. F16 has a different exponent
    ///   bias (15 vs 127) and 10 mantissa bits, so the conversion is
    ///   non-trivial — we defer to `half` for the edge cases (subnormals,
    ///   NaN propagation, infinities).
    ///
    /// `DType::FP8` returns [`LoaderError::UnsupportedDType`] — no FP8
    /// decode is planned for M4. Extending would require deciding which
    /// FP8 format (E4M3 vs E5M2) and is out of scope.
    ///
    /// The safetensors format stores all multi-byte values as
    /// little-endian regardless of host endianness, so we decode via
    /// `u16::from_le_bytes` / `f32::from_le_bytes` and never rely on
    /// `bytemuck`-style reinterpret casts.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>, LoaderError> {
        match self.dtype {
            DType::F32 => {
                if self.raw_bytes.len() % 4 != 0 {
                    return Err(LoaderError::InvalidFormat(format!(
                        "tensor '{}': F32 body length {} is not a multiple of 4",
                        self.name,
                        self.raw_bytes.len()
                    )));
                }
                let expected_elements: usize = self.shape.iter().product();
                let actual_elements = self.raw_bytes.len() / 4;
                if actual_elements != expected_elements {
                    return Err(LoaderError::InvalidFormat(format!(
                        "tensor '{}': shape {:?} implies {} F32 elements, \
                         body carries {}",
                        self.name, self.shape, expected_elements, actual_elements
                    )));
                }
                let mut out = Vec::with_capacity(actual_elements);
                for chunk in self.raw_bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(out)
            }
            DType::BF16 => {
                if self.raw_bytes.len() % 2 != 0 {
                    return Err(LoaderError::InvalidFormat(format!(
                        "tensor '{}': BF16 body length {} is not a multiple of 2",
                        self.name,
                        self.raw_bytes.len()
                    )));
                }
                let expected_elements: usize = self.shape.iter().product();
                let actual_elements = self.raw_bytes.len() / 2;
                if actual_elements != expected_elements {
                    return Err(LoaderError::InvalidFormat(format!(
                        "tensor '{}': shape {:?} implies {} BF16 elements, \
                         body carries {}",
                        self.name, self.shape, expected_elements, actual_elements
                    )));
                }
                let mut out = Vec::with_capacity(actual_elements);
                for chunk in self.raw_bytes.chunks_exact(2) {
                    let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    // BF16 occupies the upper 16 bits of an F32; the
                    // lower 16 bits are zero. This is lossless — every
                    // BF16 value maps to exactly one F32 value.
                    out.push(f32::from_bits((bf16_bits as u32) << 16));
                }
                Ok(out)
            }
            DType::F16 => {
                if self.raw_bytes.len() % 2 != 0 {
                    return Err(LoaderError::InvalidFormat(format!(
                        "tensor '{}': F16 body length {} is not a multiple of 2",
                        self.name,
                        self.raw_bytes.len()
                    )));
                }
                let expected_elements: usize = self.shape.iter().product();
                let actual_elements = self.raw_bytes.len() / 2;
                if actual_elements != expected_elements {
                    return Err(LoaderError::InvalidFormat(format!(
                        "tensor '{}': shape {:?} implies {} F16 elements, \
                         body carries {}",
                        self.name, self.shape, expected_elements, actual_elements
                    )));
                }
                let mut out = Vec::with_capacity(actual_elements);
                for chunk in self.raw_bytes.chunks_exact(2) {
                    let f16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(half::f16::from_bits(f16_bits).to_f32());
                }
                Ok(out)
            }
            DType::FP8 => Err(LoaderError::UnsupportedDType(format!(
                "tensor '{}': FP8 decode not planned for M4 scope",
                self.name
            ))),
            DType::Int8 => Err(LoaderError::UnsupportedDType(format!(
                "tensor '{}': INT8 weights are produced by the M9.1 \
                 quantizer (`tensor::quantizer::absmax_per_channel_symmetric`), \
                 not loaded directly from safetensors. Read the source \
                 BF16/F32 weight, then quantise.",
                self.name
            ))),
        }
    }
}

/// Map a `safetensors::Dtype` to Atenia's [`DType`]. Returns an error
/// message for dtypes Atenia does not carry a variant for (integer
/// types, BOOL, F64). The `safetensors` crate's enum is
/// non-exhaustive, so the `_` arm is mandatory.
fn map_dtype(st: safetensors::Dtype) -> Result<DType, String> {
    use safetensors::Dtype as St;
    match st {
        St::F32 => Ok(DType::F32),
        St::F16 => Ok(DType::F16),
        St::BF16 => Ok(DType::BF16),
        St::F8_E4M3 | St::F8_E5M2 => Ok(DType::FP8),
        other => Err(format!(
            "safetensors dtype {:?} has no Atenia DType equivalent \
             (integer, BOOL, and F64 tensors are not a Model-Loader target)",
            other
        )),
    }
}

// ---------------------------------------------------------------------------
// FP8-SAFETENSORS-1 — FP8 → F32 decoders (OCP 8-bit float formats).
// ---------------------------------------------------------------------------

/// Decode one **E4M3** byte (1 sign · 4 exp, bias 7 · 3 mantissa) to `f32`.
/// This is the `e4m3fn` variant used by safetensors / PyTorch: **no
/// infinities**, and the only NaN is `S.1111.111`. Max finite magnitude 448.
/// The conversion is exact (FP8 → F32 widening never loses bits).
fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as i32;
    if exp == 0x0F && man == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        // Subnormal: 2^(1-7) · man/8 = man · 2^-9.
        sign * (man as f32) * 2f32.powi(-9)
    } else {
        // Normal: (1 + man/8) · 2^(exp-7).
        sign * (1.0 + (man as f32) / 8.0) * 2f32.powi(exp - 7)
    }
}

/// Decode one **E5M2** byte (1 sign · 5 exp, bias 15 · 2 mantissa) to `f32`.
/// IEEE-like: `exp == 31` is ±inf (mantissa 0) or NaN. Max finite 57344.
fn fp8_e5m2_to_f32(b: u8) -> f32 {
    let neg = b & 0x80 != 0;
    let sign = if neg { -1.0f32 } else { 1.0 };
    let exp = ((b >> 2) & 0x1F) as i32;
    let man = (b & 0x03) as i32;
    if exp == 0x1F {
        return if man == 0 {
            if neg { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        };
    }
    if exp == 0 {
        // Subnormal: 2^(1-15) · man/4 = man · 2^-16.
        sign * (man as f32) * 2f32.powi(-16)
    } else {
        // Normal: (1 + man/4) · 2^(exp-15).
        sign * (1.0 + (man as f32) / 4.0) * 2f32.powi(exp - 15)
    }
}

#[cfg(test)]
mod fp8_tests {
    use super::*;

    #[test]
    fn e4m3_known_values() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0); // exp=7,man=0 → 2^0
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0); // sign + 1.0
        assert_eq!(fp8_e4m3_to_f32(0x3F), 1.875); // (1+7/8)·2^0
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0); // max finite
        assert!(fp8_e4m3_to_f32(0x7F).is_nan()); // only NaN
        assert_eq!(fp8_e4m3_to_f32(0x08), 0.015625); // 2^-6 smallest normal
    }

    #[test]
    fn e5m2_known_values() {
        assert_eq!(fp8_e5m2_to_f32(0x00), 0.0);
        assert_eq!(fp8_e5m2_to_f32(0x3C), 1.0); // exp=15,man=0
        assert_eq!(fp8_e5m2_to_f32(0xBC), -1.0);
        assert_eq!(fp8_e5m2_to_f32(0x7B), 57344.0); // max finite
        assert!(fp8_e5m2_to_f32(0x7C).is_infinite() && fp8_e5m2_to_f32(0x7C) > 0.0);
        assert!(fp8_e5m2_to_f32(0xFC).is_infinite() && fp8_e5m2_to_f32(0xFC) < 0.0);
        assert!(fp8_e5m2_to_f32(0x7D).is_nan());
    }

    #[test]
    fn fp8_widening_is_exact_roundtrippable_subset() {
        // Every E4M3 normal value is representable exactly in f32.
        for b in 0u8..=255 {
            let v = fp8_e4m3_to_f32(b);
            if v.is_finite() {
                assert_eq!(v, v as f64 as f32, "byte {b:#04x}");
            }
        }
    }
}
