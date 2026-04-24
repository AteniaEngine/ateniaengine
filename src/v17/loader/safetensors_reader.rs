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
    /// Absolute offset into `SafetensorsReader::bytes` where this
    /// tensor's payload begins.
    body_start: usize,
    /// Absolute offset (exclusive) where this tensor's payload ends.
    body_end: usize,
}

/// Owned safetensors reader. Construct via [`Self::open`] or
/// [`Self::from_bytes`]; query via [`Self::iter`], [`Self::get`], and
/// [`Self::metadata`].
#[derive(Debug)]
pub struct SafetensorsReader {
    bytes: Vec<u8>,
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
        let bytes = fs::read(path).map_err(|e| LoaderError::IoError(e.to_string()))?;
        Self::from_bytes(bytes)
    }

    /// Construct a reader from an in-memory byte buffer. Same
    /// validation as [`Self::open`] but skips the filesystem step —
    /// useful for tests and for callers that obtain bytes over the
    /// network.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LoaderError> {
        // Parse the header once to extract the optional
        // `__metadata__` map — the `SafeTensors<'data>` view produced
        // by `deserialize` does not expose a `metadata()` accessor in
        // crate version 0.4, so we call `read_metadata` separately.
        // The extra pass over the first 8 + header_len bytes is
        // negligible; the body itself is not re-scanned.
        let (_hdr_len, hdr_metadata) =
            safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| {
                LoaderError::InvalidFormat(format!(
                    "safetensors read_metadata: {:?}",
                    e
                ))
            })?;
        let metadata: Option<HashMap<String, String>> = hdr_metadata.metadata().clone();

        // `safetensors::SafeTensors::deserialize` validates the
        // header-size prefix, parses the JSON, and checks that every
        // declared tensor's `data_offsets` falls within the body.
        let view = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| {
            LoaderError::InvalidFormat(format!("safetensors deserialize: {:?}", e))
        })?;

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
        let header_len =
            u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
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

        for (name, tv) in view.tensors() {
            let dtype = map_dtype(tv.dtype()).map_err(|msg| {
                LoaderError::InvalidFormat(format!(
                    "tensor '{}': {}",
                    name, msg
                ))
            })?;

            let shape: Vec<usize> = tv.shape().to_vec();
            let data = tv.data();
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
                },
            );
            names_in_order.push(name.to_string());
        }

        // `view` is dropped here — from now on we read from `bytes`
        // directly via the offsets we captured. `metadata` was
        // already captured above from `read_metadata`.
        drop(view);

        Ok(Self {
            bytes,
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
            TensorEntry {
                name: name.as_str(),
                shape: entry.shape.as_slice(),
                dtype: entry.dtype,
                raw_bytes: &self.bytes[entry.body_start..entry.body_end],
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
        Some(TensorEntry {
            name: owned_name.as_str(),
            shape: entry.shape.as_slice(),
            dtype: entry.dtype,
            raw_bytes: &self.bytes[entry.body_start..entry.body_end],
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
    /// M4-a supports only `DType::F32`. BF16, F16, FP8 and integer
    /// dtypes return [`LoaderError::UnsupportedDType`]. M4-d extends
    /// coverage to BF16 and F16 via host-side downcast.
    ///
    /// The safetensors format stores F32 tensors as little-endian
    /// IEEE 754 bytes regardless of host endianness, so we decode via
    /// `f32::from_le_bytes` and never rely on `bytemuck`-style
    /// reinterpret casts.
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
                    out.push(f32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ]));
                }
                Ok(out)
            }
            DType::F16 => Err(LoaderError::UnsupportedDType(format!(
                "tensor '{}': F16 decode not implemented in M4-a (lands in M4-d)",
                self.name
            ))),
            DType::BF16 => Err(LoaderError::UnsupportedDType(format!(
                "tensor '{}': BF16 decode not implemented in M4-a (lands in M4-d)",
                self.name
            ))),
            DType::FP8 => Err(LoaderError::UnsupportedDType(format!(
                "tensor '{}': FP8 decode not planned for M4 scope",
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
