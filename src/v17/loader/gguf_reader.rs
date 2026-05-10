//! **M11.D.1** — GGUF format reader (header + metadata + tensor
//! descriptors only; tensor data decode lands in M11.D.2).
//!
//! GGUF (`General-Purpose GGML Unified Format`) is the
//! file format used by `llama.cpp` and the broader GGML
//! ecosystem. It supersedes the older GGML / GGJT formats and
//! is the de-facto distribution format for quantised LLM
//! checkpoints on HuggingFace (the `*-GGUF` repos).
//!
//! ## Spec
//!
//! - Authoritative spec: `ggml-org/ggml/docs/gguf.md`.
//! - Reference reader / writer: `llama.cpp/gguf-py/`.
//!
//! ## File layout (little-endian, packed, no padding)
//!
//! ```text
//! [HEADER]
//!   magic:             u32     = 0x46554747 ("GGUF" little-endian)
//!   version:           u32     ∈ {2, 3} (v1 was a wire-break-incompatible early version)
//!   tensor_count:      u64
//!   metadata_kv_count: u64
//!
//! [METADATA KV] × metadata_kv_count
//!   key:        gguf_string   (u64 length, then UTF-8 bytes, no NUL)
//!   value_type: u32           ∈ MetadataType enum (0..=12)
//!   value:      depends on value_type
//!
//! [TENSOR INFO] × tensor_count
//!   name:        gguf_string
//!   n_dims:      u32
//!   dimensions:  u64[n_dims]
//!   tensor_type: u32           (ggml_type enum)
//!   offset:      u64           (relative to data_section start)
//!
//! [PADDING TO `general.alignment` (default 32)]
//!
//! [TENSOR DATA]
//! ```
//!
//! ## Scope of this module (M11.D.1)
//!
//! Parses the file up to the end of the tensor info section,
//! computes `data_section_offset` (the absolute byte offset of
//! the first tensor's data), and exposes the metadata
//! key-value map and the tensor descriptors. The actual tensor
//! data is NOT read — that requires the per-format dequant
//! kernels (Q8_0, Q4_K, F16, ...) landing in M11.D.2.
//!
//! ## Forward compatibility
//!
//! [`GgufTensorType`] carries an `Unknown(u32)` variant so a
//! file using a quantisation type Atenia does not yet decode
//! still parses cleanly through this reader. The downstream
//! decoder (M11.D.2) is the layer that fails the load with
//! `UnsupportedDType` for the un-implemented variants.
//!
//! Versions 2 and 3 are wire-identical for little-endian files
//! (v3 only added a big-endian variant flag at byte level which
//! cannot be auto-detected). This reader accepts both versions
//! and treats them as equivalent for LE files (which is
//! ~every GGUF on HuggingFace).

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use crate::v17::loader::loader_errors::LoaderError;

/// Alias kept under the M11.D-specific name so callers can use
/// the project-typical `GgufError` while every variant is shared
/// with the rest of the loader (no duplication of error
/// infrastructure for the GGUF path).
pub type GgufError = LoaderError;

/// GGUF magic bytes spelt as a little-endian u32. The four
/// ASCII bytes 'G', 'G', 'U', 'F' read in LE order yield
/// `0x46554747`.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Default alignment for the tensor data section start and per-
/// tensor offsets when the `general.alignment` metadata key is
/// absent. Defined by the spec at 32 bytes.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Lowest GGUF wire version this reader accepts.
pub const MIN_VERSION: u32 = 2;
/// Highest GGUF wire version this reader accepts.
pub const MAX_VERSION: u32 = 3;

// ===== Metadata value-type tags =====================================

/// GGUF metadata value-type discriminator. The wire type is
/// `u32`; this enum mirrors the 13 variants defined by the
/// spec. The 4-5 gap between Int32 and Float32 reflects the
/// historical introduction of UInt64/Int64/Float64 between v1
/// and v2 — the discriminants are stable across v2/v3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MetadataType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataType {
    /// Decode a `u32` wire value into the enum. Returns
    /// `InvalidFormat` for unknown types (no `Unknown(u32)`
    /// variant — metadata-type IDs are part of the spec
    /// proper, unlike tensor-type IDs which extend
    /// independently).
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        Ok(match v {
            0 => MetadataType::UInt8,
            1 => MetadataType::Int8,
            2 => MetadataType::UInt16,
            3 => MetadataType::Int16,
            4 => MetadataType::UInt32,
            5 => MetadataType::Int32,
            6 => MetadataType::Float32,
            7 => MetadataType::Bool,
            8 => MetadataType::String,
            9 => MetadataType::Array,
            10 => MetadataType::UInt64,
            11 => MetadataType::Int64,
            12 => MetadataType::Float64,
            _ => {
                return Err(GgufError::InvalidFormat(format!(
                    "GGUF metadata: unknown value-type discriminator {} (expected 0..=12)",
                    v
                )));
            }
        })
    }
}

/// A single GGUF metadata value. `Array` is recursive —
/// nested arrays-of-arrays are legal per spec. Boolean values
/// are stored as one byte (0/1) on the wire.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    UInt64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(MetadataArray),
}

/// Element-type-tagged GGUF array. The wire format carries
/// `(element_type: u32, length: u64, values: element_type[length])`,
/// so the element type is recovered post-parse via the variant
/// of the first element (or `element_type` directly when the
/// array is empty).
#[derive(Debug, Clone)]
pub struct MetadataArray {
    /// The wire-level element type. Carried explicitly so an
    /// empty array still records what type its elements would
    /// have had.
    pub element_type: MetadataType,
    /// Decoded values. For `Array` element types, each entry
    /// is itself a `MetadataValue::Array` (nested).
    pub values: Vec<MetadataValue>,
}

impl MetadataValue {
    /// Convenience: coerce to `&str` if the value is a String.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    /// Convenience: coerce any integer variant into a u64 for
    /// dimension / count fields. Returns None for non-integer
    /// variants. Negative signed values fail the conversion.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::UInt8(v) => Some(*v as u64),
            MetadataValue::UInt16(v) => Some(*v as u64),
            MetadataValue::UInt32(v) => Some(*v as u64),
            MetadataValue::UInt64(v) => Some(*v),
            MetadataValue::Int8(v) if *v >= 0 => Some(*v as u64),
            MetadataValue::Int16(v) if *v >= 0 => Some(*v as u64),
            MetadataValue::Int32(v) if *v >= 0 => Some(*v as u64),
            MetadataValue::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    /// Convenience: coerce a Float32 / Float64 to `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            MetadataValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }
    /// Convenience: extract a Bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

// ===== Tensor type =================================================

/// Per-tensor `ggml_type` discriminator. The wire format is a
/// `u32`. Discriminants follow the canonical mapping in
/// `ggml.h`. Variants Atenia recognises by name are explicit;
/// every other (including the IQ-quant family and any future
/// formats) parses through `Unknown(u32)` so a checkpoint with
/// at least one unsupported tensor type still produces a valid
/// reader — the failure surfaces at decode time (M11.D.2) for
/// the specific tensors the engine cannot yet handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufTensorType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    /// Catch-all for tensor-type IDs Atenia does not yet name.
    /// Includes the IQ-quant family (`IQ2_XXS = 16`, `IQ4_XS = 23`,
    /// ...), the integer types (`I8`, `I16`, `I32`, `I64`), and
    /// any future format additions. The downstream decoder is
    /// responsible for failing the load with a clear error if
    /// it encounters one of these.
    Unknown(u32),
}

impl GgufTensorType {
    /// Decode a `u32` wire value. Unknown / unrecognised IDs
    /// route through `Unknown(v)` (see variant doc).
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgufTensorType::F32,
            1 => GgufTensorType::F16,
            // The canonical ggml_type ordering puts BF16 at 30.
            // This is post-IQ-quants (introduced 2024).
            30 => GgufTensorType::BF16,
            2 => GgufTensorType::Q4_0,
            3 => GgufTensorType::Q4_1,
            6 => GgufTensorType::Q5_0,
            7 => GgufTensorType::Q5_1,
            8 => GgufTensorType::Q8_0,
            9 => GgufTensorType::Q8_1,
            10 => GgufTensorType::Q2_K,
            11 => GgufTensorType::Q3_K,
            12 => GgufTensorType::Q4_K,
            13 => GgufTensorType::Q5_K,
            14 => GgufTensorType::Q6_K,
            15 => GgufTensorType::Q8_K,
            _ => GgufTensorType::Unknown(v),
        }
    }

    /// Encode back to `u32` for round-trip / debug. Unknown
    /// variants surface their captured discriminator.
    pub fn as_u32(self) -> u32 {
        match self {
            GgufTensorType::F32 => 0,
            GgufTensorType::F16 => 1,
            GgufTensorType::BF16 => 30,
            GgufTensorType::Q4_0 => 2,
            GgufTensorType::Q4_1 => 3,
            GgufTensorType::Q5_0 => 6,
            GgufTensorType::Q5_1 => 7,
            GgufTensorType::Q8_0 => 8,
            GgufTensorType::Q8_1 => 9,
            GgufTensorType::Q2_K => 10,
            GgufTensorType::Q3_K => 11,
            GgufTensorType::Q4_K => 12,
            GgufTensorType::Q5_K => 13,
            GgufTensorType::Q6_K => 14,
            GgufTensorType::Q8_K => 15,
            GgufTensorType::Unknown(v) => v,
        }
    }
}

// ===== Tensor descriptor ============================================

/// One tensor's header-level metadata. Does NOT carry the
/// tensor data — the `offset` field locates the data inside the
/// reader's `data_section_offset`-anchored region, and a
/// downstream M11.D.2 decoder reopens the file at that position
/// to materialise the values.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// GGUF naming convention (e.g. `blk.0.attn_q.weight`,
    /// `token_embd.weight`). M11.D.3 will translate to HF
    /// names for the existing graph builders.
    pub name: String,
    /// Dimensions in GGUF order. **Note**: GGUF stores 2-D
    /// linear weights as `[in_features, out_features]` —
    /// the transpose of HuggingFace safetensors's
    /// `[out_features, in_features]`. This means GGUF tensors
    /// already match Atenia's internal `[in, out]` convention,
    /// so the M11.D.3 weight mapper does NOT apply
    /// `Transpose2D` to GGUF projections.
    pub dimensions: Vec<u64>,
    /// Quantisation / dtype.
    pub tensor_type: GgufTensorType,
    /// Byte offset relative to `data_section_offset`. Add
    /// `data_section_offset` to get the absolute byte position
    /// inside the file.
    pub offset: u64,
}

// ===== Reader ======================================================

/// Parsed GGUF header. Only the descriptor section is
/// materialised — tensor data stays on disk until M11.D.2.
#[derive(Debug, Clone)]
pub struct GgufReader {
    /// GGUF wire version observed in the header (2 or 3).
    pub version: u32,
    /// Number of tensors declared by the header.
    pub tensor_count: u64,
    /// Decoded metadata key-value pairs.
    pub metadata: HashMap<String, MetadataValue>,
    /// Tensor descriptors in the order they appear in the file.
    /// Indexing by name is normally cheap because the typical
    /// GGUF carries 100s, not 10000s, of tensors; helper
    /// methods scan linearly.
    pub tensors: Vec<TensorDescriptor>,
    /// Absolute byte offset of the first tensor's data inside
    /// the file. Computed as `align_up(end_of_descriptors,
    /// alignment)` per spec.
    pub data_section_offset: u64,
    /// Effective alignment in bytes. Read from
    /// `general.alignment` if present, otherwise
    /// [`DEFAULT_ALIGNMENT`] (32).
    pub alignment: u64,
    /// Path the reader was opened from. Stored so the M11.D.2
    /// decoder can reopen the file without the caller having
    /// to thread the path through.
    pub file_path: PathBuf,
}

impl GgufReader {
    /// Parse a GGUF file's header and tensor descriptors.
    /// Tensor data is NOT read — the file is closed before
    /// this function returns.
    ///
    /// Returns `InvalidFormat` for: bad magic, unsupported
    /// version, malformed metadata KV section, malformed
    /// tensor descriptor section, dimension count of zero,
    /// arithmetic overflow on offset / alignment computation.
    pub fn read_from_path(path: &Path) -> Result<Self, GgufError> {
        let file = File::open(path).map_err(|e| {
            GgufError::IoError(format!(
                "GGUF: failed to open '{}': {e}",
                path.display()
            ))
        })?;
        // BufReader so the per-byte / per-u32 parsing does not
        // touch the syscall layer for every primitive read.
        let mut reader = BufReader::new(file);
        let mut cursor = Cursor::new(&mut reader);

        // ---- Header ----
        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidFormat(format!(
                "GGUF: bad magic 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X} = \"GGUF\" LE)"
            )));
        }
        let version = cursor.read_u32()?;
        if !(MIN_VERSION..=MAX_VERSION).contains(&version) {
            return Err(GgufError::InvalidFormat(format!(
                "GGUF: unsupported version {version} (Atenia accepts {MIN_VERSION}..={MAX_VERSION}; \
                 see ggml-org/ggml/docs/gguf.md for the version compatibility matrix)"
            )));
        }
        let tensor_count = cursor.read_u64()?;
        let metadata_kv_count = cursor.read_u64()?;

        // ---- Metadata KV section ----
        let mut metadata: HashMap<String, MetadataValue> =
            HashMap::with_capacity(metadata_kv_count as usize);
        for kv_idx in 0..metadata_kv_count {
            let key = cursor.read_gguf_string().map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "GGUF metadata KV #{kv_idx}: failed to read key: {e:?}"
                ))
            })?;
            let value_type_raw = cursor.read_u32().map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "GGUF metadata KV #{kv_idx} ('{key}'): failed to read value-type: {e:?}"
                ))
            })?;
            let value_type = MetadataType::from_u32(value_type_raw)?;
            let value = read_metadata_value(&mut cursor, value_type).map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "GGUF metadata KV #{kv_idx} ('{key}'): failed to decode value: {e:?}"
                ))
            })?;
            metadata.insert(key, value);
        }

        // ---- Tensor info section ----
        let mut tensors: Vec<TensorDescriptor> =
            Vec::with_capacity(tensor_count as usize);
        for t_idx in 0..tensor_count {
            let name = cursor.read_gguf_string().map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "GGUF tensor descriptor #{t_idx}: failed to read name: {e:?}"
                ))
            })?;
            let n_dims = cursor.read_u32()?;
            if n_dims == 0 {
                return Err(GgufError::InvalidFormat(format!(
                    "GGUF tensor '{name}': dimension count is zero"
                )));
            }
            // Spec caps n_dims at 8; reject anything obviously
            // pathological so a corrupt descriptor cannot
            // pre-allocate a Vec of u32::MAX dims.
            if n_dims > 8 {
                return Err(GgufError::InvalidFormat(format!(
                    "GGUF tensor '{name}': dimension count {n_dims} exceeds spec maximum 8"
                )));
            }
            let mut dimensions: Vec<u64> = Vec::with_capacity(n_dims as usize);
            for d_idx in 0..n_dims {
                let dim = cursor.read_u64().map_err(|e| {
                    GgufError::InvalidFormat(format!(
                        "GGUF tensor '{name}' dim {d_idx}: read failed: {e:?}"
                    ))
                })?;
                dimensions.push(dim);
            }
            let tensor_type_raw = cursor.read_u32()?;
            let tensor_type = GgufTensorType::from_u32(tensor_type_raw);
            let offset = cursor.read_u64()?;
            tensors.push(TensorDescriptor {
                name,
                dimensions,
                tensor_type,
                offset,
            });
        }

        // ---- Compute data section start ----
        //
        // The data section begins at the next multiple of
        // `general.alignment` after the end of the tensor info
        // section. The alignment is read from metadata; if
        // absent or non-positive, fall back to the spec
        // default (32).
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .filter(|a| *a > 0)
            .unwrap_or(DEFAULT_ALIGNMENT);
        let pos_after_descriptors = cursor.position();
        let data_section_offset = align_up(pos_after_descriptors, alignment).ok_or_else(
            || {
                GgufError::InvalidFormat(format!(
                    "GGUF: arithmetic overflow computing aligned data offset \
                     (post-descriptor pos {pos_after_descriptors}, alignment {alignment})"
                ))
            },
        )?;

        Ok(GgufReader {
            version,
            tensor_count,
            metadata,
            tensors,
            data_section_offset,
            alignment,
            file_path: path.to_path_buf(),
        })
    }

    /// Convenience: lookup a tensor descriptor by GGUF name.
    /// Linear scan; sufficient for the ≤ 1000-tensor regime
    /// every Llama / Phi-3 / Gemma 2 GGUF lives in.
    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Convenience: read `general.architecture` as a string.
    /// Returns `None` if the key is absent or not a string.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_string())
    }
}

// ===== Internals: byte cursor + value parsers =======================

/// Thin sequential reader over a `BufReader<File>` that tracks
/// byte position. Every primitive read is little-endian per
/// spec.
struct Cursor<'a> {
    inner: &'a mut BufReader<File>,
    pos: u64,
}

impl<'a> Cursor<'a> {
    fn new(inner: &'a mut BufReader<File>) -> Self {
        Self { inner, pos: 0 }
    }
    fn position(&self) -> u64 {
        self.pos
    }
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), GgufError> {
        self.inner.read_exact(buf).map_err(|e| {
            GgufError::IoError(format!(
                "GGUF: read failed at byte {} (need {} bytes): {e}",
                self.pos,
                buf.len()
            ))
        })?;
        self.pos += buf.len() as u64;
        Ok(())
    }
    fn read_u8(&mut self) -> Result<u8, GgufError> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }
    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(self.read_u16()? as i16)
    }
    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.read_u32()? as i32)
    }
    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.read_u64()? as i64)
    }
    fn read_f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.read_u32()?))
    }
    fn read_f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.read_u64()?))
    }
    fn read_bool(&mut self) -> Result<bool, GgufError> {
        // Spec: one-byte boolean; 0 = false, anything else
        // = true. Reject non-canonical values to surface
        // file corruption early.
        let b = self.read_u8()?;
        match b {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(GgufError::InvalidFormat(format!(
                "GGUF: non-canonical bool byte 0x{other:02X} (expected 0x00 or 0x01)"
            ))),
        }
    }
    /// Read a `gguf_string`: u64 length, then UTF-8 bytes.
    fn read_gguf_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()?;
        // Defensive cap: a metadata string longer than 64 MiB
        // is almost certainly a parse-state desync. Reject
        // before allocating.
        const STRING_LEN_CAP: u64 = 64 * 1024 * 1024;
        if len > STRING_LEN_CAP {
            return Err(GgufError::InvalidFormat(format!(
                "GGUF: string length {len} exceeds {STRING_LEN_CAP}-byte sanity cap"
            )));
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| {
            GgufError::InvalidFormat(format!("GGUF: string is not valid UTF-8: {e}"))
        })
    }
}

/// Decode one metadata value of declared type. Recursive on
/// `MetadataType::Array`.
fn read_metadata_value(
    c: &mut Cursor,
    value_type: MetadataType,
) -> Result<MetadataValue, GgufError> {
    Ok(match value_type {
        MetadataType::UInt8 => MetadataValue::UInt8(c.read_u8()?),
        MetadataType::Int8 => MetadataValue::Int8(c.read_i8()?),
        MetadataType::UInt16 => MetadataValue::UInt16(c.read_u16()?),
        MetadataType::Int16 => MetadataValue::Int16(c.read_i16()?),
        MetadataType::UInt32 => MetadataValue::UInt32(c.read_u32()?),
        MetadataType::Int32 => MetadataValue::Int32(c.read_i32()?),
        MetadataType::Float32 => MetadataValue::Float32(c.read_f32()?),
        MetadataType::Bool => MetadataValue::Bool(c.read_bool()?),
        MetadataType::String => MetadataValue::String(c.read_gguf_string()?),
        MetadataType::UInt64 => MetadataValue::UInt64(c.read_u64()?),
        MetadataType::Int64 => MetadataValue::Int64(c.read_i64()?),
        MetadataType::Float64 => MetadataValue::Float64(c.read_f64()?),
        MetadataType::Array => {
            let element_type_raw = c.read_u32()?;
            let element_type = MetadataType::from_u32(element_type_raw)?;
            let len = c.read_u64()?;
            // Defensive cap: a metadata array longer than
            // 256 M elements is almost certainly a parse-state
            // desync (the largest legitimate array is the
            // tokeniser vocab — Gemma 2's 256 K tokens fits
            // comfortably under this cap).
            const ARRAY_LEN_CAP: u64 = 256 * 1024 * 1024;
            if len > ARRAY_LEN_CAP {
                return Err(GgufError::InvalidFormat(format!(
                    "GGUF: array length {len} exceeds {ARRAY_LEN_CAP}-element sanity cap"
                )));
            }
            let mut values: Vec<MetadataValue> = Vec::with_capacity(len as usize);
            for _ in 0..len {
                values.push(read_metadata_value(c, element_type)?);
            }
            MetadataValue::Array(MetadataArray {
                element_type,
                values,
            })
        }
    })
}

/// Round `pos` up to the next multiple of `alignment`. Returns
/// `None` on arithmetic overflow.
fn align_up(pos: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 {
        return Some(pos);
    }
    let rem = pos % alignment;
    if rem == 0 {
        return Some(pos);
    }
    pos.checked_add(alignment - rem)
}

// ===== Tests =======================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// `align_up` rounds a non-aligned offset up to the next
    /// multiple, leaves an already-aligned offset untouched,
    /// and detects overflow.
    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(31, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(33, 32), Some(64));
        assert_eq!(align_up(u64::MAX - 1, 32), None);
    }

    /// `MetadataType::from_u32` rejects out-of-range
    /// discriminators with a clear message.
    #[test]
    fn metadata_type_rejects_unknown_discriminator() {
        let err = MetadataType::from_u32(99).expect_err("99 must fail");
        match err {
            GgufError::InvalidFormat(msg) => {
                assert!(msg.contains("99"), "message should mention the value: {msg}");
                assert!(msg.contains("0..=12"));
            }
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    /// `GgufTensorType::from_u32` round-trips known formats
    /// and routes unknown discriminators through `Unknown(v)`.
    #[test]
    fn tensor_type_round_trip_and_unknown() {
        let known = [
            (0u32, GgufTensorType::F32),
            (1, GgufTensorType::F16),
            (8, GgufTensorType::Q8_0),
            (12, GgufTensorType::Q4_K),
            (14, GgufTensorType::Q6_K),
            (30, GgufTensorType::BF16),
        ];
        for (raw, ty) in known {
            assert_eq!(GgufTensorType::from_u32(raw), ty);
            assert_eq!(ty.as_u32(), raw);
        }
        // The IQ-quant family lives at 16..=23 (post-2024).
        // Atenia does not name them yet — they must route
        // through Unknown.
        for raw in [16u32, 23, 24, 100, 999] {
            assert_eq!(GgufTensorType::from_u32(raw), GgufTensorType::Unknown(raw));
        }
    }

    /// **M11.D.1** — read the production TinyLlama
    /// 1.1B-Chat-Q8_0 GGUF and confirm header / metadata /
    /// tensor descriptors parse coherently.
    ///
    /// Skips itself when the local GGUF fixture is absent.
    #[test]
    fn reads_tinyllama_q8_0() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/tinyllama-q8_0/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");
        if !path.exists() {
            eprintln!(
                "[skip] TinyLlama GGUF not found at {}; download via \
                 huggingface_hub `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`",
                path.display()
            );
            return;
        }
        let r = GgufReader::read_from_path(&path).expect("GGUF must parse");
        assert!(
            r.version == 2 || r.version == 3,
            "expected version 2 or 3, got {}",
            r.version
        );
        assert!(r.tensor_count > 0, "tensor_count must be positive");
        assert_eq!(
            r.tensors.len(),
            r.tensor_count as usize,
            "decoded descriptors should match declared count"
        );
        // Architecture must be llama.
        assert_eq!(
            r.architecture(),
            Some("llama"),
            "general.architecture must be 'llama' for TinyLlama"
        );
        // TinyLlama 1.1B has 22 decoder layers.
        let block_count = r
            .metadata
            .get("llama.block_count")
            .and_then(|v| v.as_u64())
            .expect("llama.block_count must be present and integer-valued");
        assert_eq!(block_count, 22, "TinyLlama has 22 layers");
        // Token embedding tensor must exist with type Q8_0.
        let embd = r
            .tensor_by_name("token_embd.weight")
            .expect("token_embd.weight must be present");
        assert_eq!(
            embd.tensor_type,
            GgufTensorType::Q8_0,
            "TinyLlama-Q8_0 must store token_embd as Q8_0"
        );
        // Two-dimensional, [vocab, hidden] in GGUF order.
        assert_eq!(
            embd.dimensions.len(),
            2,
            "embedding tensor must be rank-2"
        );
        // Data section start is aligned.
        assert_eq!(
            r.data_section_offset % r.alignment,
            0,
            "data_section_offset must be a multiple of alignment"
        );
        // Per-tensor offset is also aligned.
        for t in &r.tensors {
            assert_eq!(
                t.offset % r.alignment,
                0,
                "tensor '{}' offset {} not aligned to {}",
                t.name,
                t.offset,
                r.alignment
            );
        }
    }

    /// Bad magic produces a clear error.
    #[test]
    fn bad_magic_rejected() {
        // Construct a tiny invalid GGUF in a tempdir.
        let tmp = std::env::temp_dir().join("atenia_gguf_bad_magic.gguf");
        std::fs::write(&tmp, b"\x00\x00\x00\x00").expect("write tmp");
        let err = GgufReader::read_from_path(&tmp).expect_err("must fail");
        let _ = std::fs::remove_file(&tmp);
        match err {
            GgufError::InvalidFormat(msg) => {
                assert!(msg.contains("magic"), "msg should mention magic: {msg}");
            }
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    /// Bad version produces a clear error.
    #[test]
    fn bad_version_rejected() {
        // Magic OK + version = 1 (pre-v2, unsupported).
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
        let tmp = std::env::temp_dir().join("atenia_gguf_bad_version.gguf");
        std::fs::write(&tmp, &bytes).expect("write tmp");
        let err = GgufReader::read_from_path(&tmp).expect_err("must fail");
        let _ = std::fs::remove_file(&tmp);
        match err {
            GgufError::InvalidFormat(msg) => {
                assert!(
                    msg.contains("version") && msg.contains("1"),
                    "msg should mention version and the bad value: {msg}"
                );
            }
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    /// Empty header (zero tensors, zero metadata, version 3)
    /// parses successfully and yields a reader with the
    /// expected default alignment.
    #[test]
    fn minimal_empty_header_parses() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
        let tmp = std::env::temp_dir().join("atenia_gguf_empty.gguf");
        std::fs::write(&tmp, &bytes).expect("write tmp");
        let r = GgufReader::read_from_path(&tmp).expect("empty header parses");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(r.version, 3);
        assert_eq!(r.tensor_count, 0);
        assert!(r.tensors.is_empty());
        assert!(r.metadata.is_empty());
        assert_eq!(r.alignment, DEFAULT_ALIGNMENT);
        // Post-header position is 24 bytes (4 + 4 + 8 + 8).
        // Aligned to 32 → data_section_offset = 32.
        assert_eq!(r.data_section_offset, 32);
    }

    /// Header with a string metadata key-value parses and the
    /// value comes back through the `as_string` accessor.
    #[test]
    fn string_metadata_parses() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata_count
        // key: "general.architecture"
        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        // value_type = String (8)
        bytes.extend_from_slice(&8u32.to_le_bytes());
        // value: "llama"
        let val = b"llama";
        bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
        bytes.extend_from_slice(val);
        let tmp = std::env::temp_dir().join("atenia_gguf_string_meta.gguf");
        std::fs::write(&tmp, &bytes).expect("write tmp");
        let r = GgufReader::read_from_path(&tmp).expect("parses");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(r.architecture(), Some("llama"));
    }

    /// Array metadata round-trips with the element type tagged
    /// correctly. Uses an empty `UInt32` array because that
    /// exercises the element-type encoding without needing
    /// any element bytes.
    #[test]
    fn empty_array_metadata_parses() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let key = b"foo";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        // value_type = Array (9)
        bytes.extend_from_slice(&9u32.to_le_bytes());
        // element_type = UInt32 (4)
        bytes.extend_from_slice(&4u32.to_le_bytes());
        // length = 0
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let tmp = std::env::temp_dir().join("atenia_gguf_empty_array.gguf");
        std::fs::write(&tmp, &bytes).expect("write tmp");
        let r = GgufReader::read_from_path(&tmp).expect("parses");
        let _ = std::fs::remove_file(&tmp);
        match r.metadata.get("foo").expect("foo present") {
            MetadataValue::Array(arr) => {
                assert_eq!(arr.element_type, MetadataType::UInt32);
                assert!(arr.values.is_empty());
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }
}
