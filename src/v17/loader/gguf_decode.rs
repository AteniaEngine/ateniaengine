//! **M11.D.2** — Per-tensor decode for GGUF files.
//!
//! Given a [`GgufReader`] (M11.D.1) and a [`TensorDescriptor`],
//! materialise the tensor values as `Vec<f32>` in CPU. The
//! result enters the existing
//! [`crate::v17::loader::weight_mapper::LoadTransform`]
//! pipeline as if it had come from a safetensors file —
//! the rest of the load path (Transpose2D / TileGroupedDim /
//! Scale / Reshape / AddScalar, and the optional BF16
//! down-convert) is bit-equivalent.
//!
//! ## Scope of this module (M11.D.2)
//!
//! Two formats:
//!
//! - **F16** — every two bytes is a little-endian IEEE-754
//!   half. Decoded via `half::f16::from_bits(...).to_f32()`,
//!   the same pattern the safetensors reader already uses
//!   (`src/v17/loader/safetensors_reader.rs:351`).
//!
//! - **Q8_0** — 32-element blocks of 34 bytes each:
//!
//!   ```text
//!     struct block_q8_0 {
//!         d: f16,           // scale, 2 bytes LE
//!         qs: [i8; 32],    // quantised values
//!     };  // sizeof = 34
//!     y[i] = (d as f32) * (qs[i] as f32)   // no zero-point
//!   ```
//!
//!   Numel must be a multiple of 32; partial blocks are
//!   undefined under spec. The decoder fails fast with
//!   `InvalidFormat` if numel % 32 != 0.
//!
//! ## What this module deliberately does NOT do
//!
//! - **Q4_0 / Q4_K / Q5_K / Q6_K / IQ-quants** land in
//!   subsequent M11.D milestones (Q4_K_M is the real
//!   production unlock per the M11.D investigation; the
//!   M11.D.MVP plan is Q8_0 + F16 only).
//!
//! - **Re-quantising the Q8_0 stream into the M9.4 INT8
//!   layout for VRAM-dense storage** is a separate optional
//!   path gated by `ATENIA_M9_INT8=1`; it is NOT implemented
//!   here. The M11.D.MVP path decodes Q8_0 → F32 → standard
//!   load pipeline → eventual BF16 down-convert (or F32
//!   resident, depending on the tier-aware loader's
//!   placement).
//!
//! - **Tensor-name translation** (GGUF `blk.0.attn_q.weight`
//!   → HuggingFace `model.layers.0.self_attn.q_proj.weight`)
//!   lands in M11.D.3.

#![allow(non_camel_case_types)]

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::v17::loader::gguf_reader::{GgufError, GgufReader, GgufTensorType, TensorDescriptor};

/// Q8_0 block size in elements, per spec (`QK8_0`).
pub const Q8_0_BLOCK_ELEMS: usize = 32;

/// Q8_0 block size in bytes: 2-byte f16 scale + 32 × i8.
pub const Q8_0_BLOCK_BYTES: usize = 2 + Q8_0_BLOCK_ELEMS;

pub const Q4_K_BLOCK_ELEMS: usize = 256;
pub const Q4_K_SCALE_BYTES: usize = 12;
pub const Q4_K_QUANT_BYTES: usize = Q4_K_BLOCK_ELEMS / 2;
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + Q4_K_SCALE_BYTES + Q4_K_QUANT_BYTES;
pub const Q6_K_BLOCK_ELEMS: usize = 256;
pub const Q6_K_QL_BYTES: usize = Q6_K_BLOCK_ELEMS / 2;
pub const Q6_K_QH_BYTES: usize = Q6_K_BLOCK_ELEMS / 4;
pub const Q6_K_SCALE_BYTES: usize = Q6_K_BLOCK_ELEMS / 16;
pub const Q6_K_BLOCK_BYTES: usize = Q6_K_QL_BYTES + Q6_K_QH_BYTES + Q6_K_SCALE_BYTES + 2;
// Q5_K (`block_q5_K`, QK_K = 256): f16 `d` + f16 `dmin` +
// 12-byte 6-bit packed scales/mins (identical layout to Q4_K,
// shared `get_scale_min_k4`) + 32-byte `qh` (1 high bit per
// weight) + 128-byte `qs` (4 low bits per weight). 176 B / 256.
pub const Q5_K_BLOCK_ELEMS: usize = 256;
pub const Q5_K_SCALE_BYTES: usize = 12;
pub const Q5_K_QH_BYTES: usize = Q5_K_BLOCK_ELEMS / 8;
pub const Q5_K_QS_BYTES: usize = Q5_K_BLOCK_ELEMS / 2;
pub const Q5_K_BLOCK_BYTES: usize =
    2 + 2 + Q5_K_SCALE_BYTES + Q5_K_QH_BYTES + Q5_K_QS_BYTES;

/// Decode a tensor's data into a fresh `Vec<f32>` in row-major
/// contiguous layout. Reopens the file and seeks to
/// `reader.data_section_offset + descriptor.offset`.
///
/// Returns:
/// - `Ok(Vec<f32>)` of length `product(descriptor.dimensions)`
///   on success.
/// - `Err(InvalidFormat)` for: rank zero, dimension overflow,
///   numel-not-multiple-of-block-size, short reads.
/// - `Err(UnsupportedDType)` for tensor types this module
///   does not yet decode (Q4_*, Q5_*, Q6_*, K-quants,
///   IQ-quants, integers, BF16, ...).
pub fn decode_tensor(
    reader: &GgufReader,
    descriptor: &TensorDescriptor,
) -> Result<Vec<f32>, GgufError> {
    let numel = compute_numel(descriptor)?;

    // Open the file and seek to the tensor's absolute byte
    // position. The reader stored `file_path` precisely so the
    // decoder can do this without the caller threading a path
    // through.
    let file = File::open(&reader.file_path).map_err(|e| {
        GgufError::IoError(format!(
            "GGUF decode '{}': failed to reopen '{}': {e}",
            descriptor.name,
            reader.file_path.display()
        ))
    })?;
    let mut buf = BufReader::new(file);
    let abs_offset = reader
        .data_section_offset
        .checked_add(descriptor.offset)
        .ok_or_else(|| {
            GgufError::InvalidFormat(format!(
                "GGUF decode '{}': arithmetic overflow on \
                 (data_section={} + offset={})",
                descriptor.name, reader.data_section_offset, descriptor.offset
            ))
        })?;
    buf.seek(SeekFrom::Start(abs_offset)).map_err(|e| {
        GgufError::IoError(format!(
            "GGUF decode '{}': seek to {abs_offset} failed: {e}",
            descriptor.name
        ))
    })?;

    match descriptor.tensor_type {
        GgufTensorType::F32 => decode_f32(&mut buf, numel, &descriptor.name),
        GgufTensorType::F16 => decode_f16(&mut buf, numel, &descriptor.name),
        GgufTensorType::Q8_0 => decode_q8_0(&mut buf, numel, &descriptor.name),
        GgufTensorType::Q4_K => decode_q4_k(&mut buf, numel, &descriptor.name),
        GgufTensorType::Q5_K => decode_q5_k(&mut buf, numel, &descriptor.name),
        GgufTensorType::Q6_K => decode_q6_k(&mut buf, numel, &descriptor.name),
        other => Err(GgufError::UnsupportedDType(format!(
            "GGUF decode '{}': tensor type {:?} (raw {}) is not implemented \
             in M11.D.2 (F16 + Q8_0 only). Q4_K_M and friends land in \
             subsequent M11.D milestones.",
            descriptor.name,
            other,
            other.as_u32()
        ))),
    }
}

/// Compute the element count from a descriptor's dimensions.
/// All-zero-dim tensors have numel zero (legal for empty
/// vocab placeholders, etc.); a zero individual dim with a
/// non-zero rank still yields zero. Overflow on the
/// multiplication is reported clearly.
fn compute_numel(descriptor: &TensorDescriptor) -> Result<usize, GgufError> {
    let mut numel: u64 = 1;
    for &d in &descriptor.dimensions {
        numel = numel.checked_mul(d).ok_or_else(|| {
            GgufError::InvalidFormat(format!(
                "GGUF decode '{}': numel overflow on dims {:?}",
                descriptor.name, descriptor.dimensions
            ))
        })?;
    }
    if numel > usize::MAX as u64 {
        return Err(GgufError::InvalidFormat(format!(
            "GGUF decode '{}': numel {numel} exceeds usize::MAX on this platform",
            descriptor.name
        )));
    }
    Ok(numel as usize)
}

/// F32 decoder: read `numel * 4` bytes, transmute as
/// little-endian f32. (On every platform Atenia targets,
/// f32 is LE; the read is byte-for-byte then bit-cast.)
fn decode_f32(buf: &mut BufReader<File>, numel: usize, name: &str) -> Result<Vec<f32>, GgufError> {
    let mut bytes = vec![0u8; numel * 4];
    buf.read_exact(&mut bytes).map_err(|e| {
        GgufError::IoError(format!(
            "GGUF decode '{name}': F32 read of {} bytes failed: {e}",
            bytes.len()
        ))
    })?;
    let mut out: Vec<f32> = Vec::with_capacity(numel);
    for chunk in bytes.chunks_exact(4) {
        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

/// F16 decoder: read `numel * 2` bytes; decode each LE u16
/// as a `half::f16` and convert to f32 via the crate's
/// IEEE-754-bit-exact `to_f32()`.
fn decode_f16(buf: &mut BufReader<File>, numel: usize, name: &str) -> Result<Vec<f32>, GgufError> {
    let mut bytes = vec![0u8; numel * 2];
    buf.read_exact(&mut bytes).map_err(|e| {
        GgufError::IoError(format!(
            "GGUF decode '{name}': F16 read of {} bytes failed: {e}",
            bytes.len()
        ))
    })?;
    let mut out: Vec<f32> = Vec::with_capacity(numel);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(half::f16::from_bits(bits).to_f32());
    }
    Ok(out)
}

/// Q8_0 decoder: 32-element blocks of (f16 scale, 32 × i8).
/// Output element `y[i] = (scale as f32) * (qs[i] as f32)`,
/// no zero-point. Numel must be a multiple of 32.
///
/// The block stride is 34 bytes — a single `read_exact` of
/// `numel/32 * 34` bytes amortises the syscall cost across
/// the whole tensor, then the inner loop is a pure CPU pass.
/// For TinyLlama 1.1B's `token_embd.weight`
/// (32000 × 2048 = 65 536 000 elements → 2 048 000 blocks
/// → 69 632 000 bytes) the read is a single ~70 MB chunk.
fn decode_q8_0(buf: &mut BufReader<File>, numel: usize, name: &str) -> Result<Vec<f32>, GgufError> {
    if numel % Q8_0_BLOCK_ELEMS != 0 {
        return Err(GgufError::InvalidFormat(format!(
            "GGUF decode '{name}': Q8_0 numel {numel} is not a multiple of \
             block size {Q8_0_BLOCK_ELEMS}; partial blocks are not defined \
             under the GGUF spec"
        )));
    }
    let n_blocks = numel / Q8_0_BLOCK_ELEMS;
    let total_bytes = n_blocks * Q8_0_BLOCK_BYTES;
    let mut bytes = vec![0u8; total_bytes];
    buf.read_exact(&mut bytes).map_err(|e| {
        GgufError::IoError(format!(
            "GGUF decode '{name}': Q8_0 read of {total_bytes} bytes \
             ({n_blocks} blocks × {Q8_0_BLOCK_BYTES} B) failed: {e}"
        ))
    })?;

    let mut out: Vec<f32> = Vec::with_capacity(numel);
    for b in 0..n_blocks {
        let block_off = b * Q8_0_BLOCK_BYTES;
        // Bytes 0..2: f16 scale (little-endian).
        let scale_bits = u16::from_le_bytes([bytes[block_off], bytes[block_off + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        // Bytes 2..34: 32 × i8 quantised values.
        let qs_off = block_off + 2;
        for k in 0..Q8_0_BLOCK_ELEMS {
            let q = bytes[qs_off + k] as i8;
            out.push(scale * (q as f32));
        }
    }
    Ok(out)
}

fn decode_q4_k(buf: &mut BufReader<File>, numel: usize, name: &str) -> Result<Vec<f32>, GgufError> {
    if numel % Q4_K_BLOCK_ELEMS != 0 {
        return Err(GgufError::InvalidFormat(format!(
            "GGUF decode '{name}': Q4_K numel {numel} is not a multiple of \
             block size {Q4_K_BLOCK_ELEMS}; partial blocks are not defined \
             under the GGUF spec"
        )));
    }
    let n_blocks = numel / Q4_K_BLOCK_ELEMS;
    let total_bytes = n_blocks * Q4_K_BLOCK_BYTES;
    let mut bytes = vec![0u8; total_bytes];
    buf.read_exact(&mut bytes).map_err(|e| {
        GgufError::IoError(format!(
            "GGUF decode '{name}': Q4_K read of {total_bytes} bytes \
             ({n_blocks} blocks × {Q4_K_BLOCK_BYTES} B) failed: {e}"
        ))
    })?;

    let mut out: Vec<f32> = Vec::with_capacity(numel);
    for b in 0..n_blocks {
        let block_off = b * Q4_K_BLOCK_BYTES;
        let d = half::f16::from_bits(u16::from_le_bytes([bytes[block_off], bytes[block_off + 1]]))
            .to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([
            bytes[block_off + 2],
            bytes[block_off + 3],
        ]))
        .to_f32();
        let scales = &bytes[block_off + 4..block_off + 4 + Q4_K_SCALE_BYTES];
        let mut qs = block_off + 4 + Q4_K_SCALE_BYTES;
        let mut is = 0usize;
        for _ in (0..Q4_K_BLOCK_ELEMS).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * (sc1 as f32);
            let min1 = dmin * (m1 as f32);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * (sc2 as f32);
            let min2 = dmin * (m2 as f32);
            for l in 0..32 {
                out.push(d1 * ((bytes[qs + l] & 0x0F) as f32) - min1);
            }
            for l in 0..32 {
                out.push(d2 * ((bytes[qs + l] >> 4) as f32) - min2);
            }
            qs += 32;
            is += 2;
        }
    }
    Ok(out)
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn decode_q6_k(buf: &mut BufReader<File>, numel: usize, name: &str) -> Result<Vec<f32>, GgufError> {
    if numel % Q6_K_BLOCK_ELEMS != 0 {
        return Err(GgufError::InvalidFormat(format!(
            "GGUF decode '{name}': Q6_K numel {numel} is not a multiple of \
             block size {Q6_K_BLOCK_ELEMS}; partial blocks are not defined \
             under the GGUF spec"
        )));
    }
    let n_blocks = numel / Q6_K_BLOCK_ELEMS;
    let total_bytes = n_blocks * Q6_K_BLOCK_BYTES;
    let mut bytes = vec![0u8; total_bytes];
    buf.read_exact(&mut bytes).map_err(|e| {
        GgufError::IoError(format!(
            "GGUF decode '{name}': Q6_K read of {total_bytes} bytes \
             ({n_blocks} blocks × {Q6_K_BLOCK_BYTES} B) failed: {e}"
        ))
    })?;

    let mut out = vec![0.0_f32; numel];
    for b in 0..n_blocks {
        let block_off = b * Q6_K_BLOCK_BYTES;
        let ql_off = block_off;
        let qh_off = ql_off + Q6_K_QL_BYTES;
        let sc_off = qh_off + Q6_K_QH_BYTES;
        let d_off = sc_off + Q6_K_SCALE_BYTES;
        let d = half::f16::from_bits(u16::from_le_bytes([bytes[d_off], bytes[d_off + 1]])).to_f32();
        let base_out = b * Q6_K_BLOCK_ELEMS;
        for n in (0..Q6_K_BLOCK_ELEMS).step_by(128) {
            let ql = ql_off + (n / 128) * 64;
            let qh = qh_off + (n / 128) * 32;
            let sc = sc_off + (n / 128) * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 =
                    (((bytes[ql + l] & 0x0F) | (((bytes[qh + l] >> 0) & 0x03) << 4)) as i8) - 32;
                let q2 = (((bytes[ql + l + 32] & 0x0F) | (((bytes[qh + l] >> 2) & 0x03) << 4))
                    as i8)
                    - 32;
                let q3 = (((bytes[ql + l] >> 4) | (((bytes[qh + l] >> 4) & 0x03) << 4)) as i8) - 32;
                let q4 =
                    (((bytes[ql + l + 32] >> 4) | (((bytes[qh + l] >> 6) & 0x03) << 4)) as i8) - 32;
                out[base_out + n + l] = d * (bytes[sc + is] as i8 as f32) * (q1 as f32);
                out[base_out + n + l + 32] = d * (bytes[sc + is + 2] as i8 as f32) * (q2 as f32);
                out[base_out + n + l + 64] = d * (bytes[sc + is + 4] as i8 as f32) * (q3 as f32);
                out[base_out + n + l + 96] = d * (bytes[sc + is + 6] as i8 as f32) * (q4 as f32);
            }
        }
    }
    Ok(out)
}

/// Q5_K decoder. Mirrors `decode_q4_k` (same f16 `d`/`dmin`, same
/// 12-byte 6-bit `get_scale_min_k4` packing, same per-64-element
/// scale/min split) plus the `qh` high-bit plane handled exactly
/// as upstream llama.cpp `dequantize_row_q5_K`: the quant value is
/// `(qs nibble) + (16 if the qh bit is set)`, with the qh bit
/// selector shifting left by 2 every 64-element group.
fn decode_q5_k(buf: &mut BufReader<File>, numel: usize, name: &str) -> Result<Vec<f32>, GgufError> {
    if numel % Q5_K_BLOCK_ELEMS != 0 {
        return Err(GgufError::InvalidFormat(format!(
            "GGUF decode '{name}': Q5_K numel {numel} is not a multiple of \
             block size {Q5_K_BLOCK_ELEMS}; partial blocks are not defined \
             under the GGUF spec"
        )));
    }
    let n_blocks = numel / Q5_K_BLOCK_ELEMS;
    let total_bytes = n_blocks * Q5_K_BLOCK_BYTES;
    let mut bytes = vec![0u8; total_bytes];
    buf.read_exact(&mut bytes).map_err(|e| {
        GgufError::IoError(format!(
            "GGUF decode '{name}': Q5_K read of {total_bytes} bytes \
             ({n_blocks} blocks × {Q5_K_BLOCK_BYTES} B) failed: {e}"
        ))
    })?;

    let mut out: Vec<f32> = Vec::with_capacity(numel);
    for b in 0..n_blocks {
        let block_off = b * Q5_K_BLOCK_BYTES;
        let d = half::f16::from_bits(u16::from_le_bytes([bytes[block_off], bytes[block_off + 1]]))
            .to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([
            bytes[block_off + 2],
            bytes[block_off + 3],
        ]))
        .to_f32();
        let scales = &bytes[block_off + 4..block_off + 4 + Q5_K_SCALE_BYTES];
        let qh_off = block_off + 4 + Q5_K_SCALE_BYTES;
        let mut qs = qh_off + Q5_K_QH_BYTES;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in (0..Q5_K_BLOCK_ELEMS).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * (sc1 as f32);
            let min1 = dmin * (m1 as f32);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * (sc2 as f32);
            let min2 = dmin * (m2 as f32);
            for l in 0..32 {
                let hi = if bytes[qh_off + l] & u1 != 0 { 16.0 } else { 0.0 };
                out.push(d1 * (((bytes[qs + l] & 0x0F) as f32) + hi) - min1);
            }
            for l in 0..32 {
                let hi = if bytes[qh_off + l] & u2 != 0 { 16.0 } else { 0.0 };
                out.push(d2 * (((bytes[qs + l] >> 4) as f32) + hi) - min2);
            }
            qs += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

// ===== Tests ========================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v17::loader::gguf_reader::{GgufReader, TensorDescriptor};
    use std::path::PathBuf;

    /// Synthetic Q8_0 block: scale=1.0, qs=[0,1,2,...,31].
    /// Decoded values must equal [0.0, 1.0, 2.0, ..., 31.0].
    fn build_q8_0_block(scale: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Q8_0_BLOCK_BYTES);
        let scale_f16 = half::f16::from_f32(scale);
        bytes.extend_from_slice(&scale_f16.to_bits().to_le_bytes());
        for &q in qs {
            bytes.push(q as u8);
        }
        assert_eq!(bytes.len(), Q8_0_BLOCK_BYTES);
        bytes
    }

    fn set_scale_min_k4(j: usize, q: &mut [u8; Q4_K_SCALE_BYTES], d: u8, m: u8) {
        assert!(d < 64);
        assert!(m < 64);
        if j < 4 {
            q[j] = (q[j] & 0xC0) | d;
            q[j + 4] = (q[j + 4] & 0xC0) | m;
        } else {
            q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
            q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
            q[j] = (q[j] & 0x3F) | ((m >> 4) << 6);
        }
    }

    fn build_q4_k_block(d: f32, dmin: f32, scales_mins: &[(u8, u8); 8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Q4_K_BLOCK_BYTES);
        bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        bytes.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        let mut scales = [0u8; Q4_K_SCALE_BYTES];
        for (j, &(scale, min)) in scales_mins.iter().enumerate() {
            set_scale_min_k4(j, &mut scales, scale, min);
        }
        bytes.extend_from_slice(&scales);
        for j in 0..(Q4_K_BLOCK_ELEMS / 64) {
            for l in 0..32 {
                let low = ((l + j) & 0x0F) as u8;
                let high = (15usize.wrapping_sub(l).wrapping_add(j) & 0x0F) as u8;
                bytes.push(low | (high << 4));
            }
        }
        assert_eq!(bytes.len(), Q4_K_BLOCK_BYTES);
        bytes
    }

    /// One synthetic Q5_K block. Same scale/min packing as Q4_K
    /// (`set_scale_min_k4`, 12 bytes) plus the 32-byte `qh`
    /// high-bit plane and 128-byte `qs` low-nibble plane.
    /// Deterministic content: `qh[l] = l`, `qs[i] = i`.
    fn build_q5_k_block(d: f32, dmin: f32, scales_mins: &[(u8, u8); 8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Q5_K_BLOCK_BYTES);
        bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        bytes.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        let mut scales = [0u8; Q4_K_SCALE_BYTES];
        for (j, &(scale, min)) in scales_mins.iter().enumerate() {
            set_scale_min_k4(j, &mut scales, scale, min);
        }
        bytes.extend_from_slice(&scales);
        for l in 0..Q5_K_QH_BYTES {
            bytes.push(l as u8);
        }
        for i in 0..Q5_K_QS_BYTES {
            bytes.push(i as u8);
        }
        assert_eq!(bytes.len(), Q5_K_BLOCK_BYTES);
        bytes
    }

    /// Helper: build a one-tensor in-memory GGUF and write it
    /// to a temp file, then return the parsed reader so the
    /// test can drive `decode_tensor` against a fully
    /// round-tripped artefact.
    fn write_minimal_gguf_and_open(
        tensor_name: &str,
        tensor_type: GgufTensorType,
        dims: &[u64],
        tensor_data: &[u8],
        tmp_filename: &str,
    ) -> (GgufReader, TensorDescriptor) {
        use crate::v17::loader::gguf_reader::GGUF_MAGIC;
        let mut bytes: Vec<u8> = Vec::new();
        // Header.
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
        // Tensor descriptor.
        bytes.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(tensor_name.as_bytes());
        bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        bytes.extend_from_slice(&tensor_type.as_u32().to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
        // Pad to alignment 32.
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        // Tensor data.
        bytes.extend_from_slice(tensor_data);
        // Write & open.
        let tmp = std::env::temp_dir().join(tmp_filename);
        std::fs::write(&tmp, &bytes).expect("write tmp");
        let reader = GgufReader::read_from_path(&tmp).expect("parse");
        let desc = reader
            .tensor_by_name(tensor_name)
            .expect("descriptor present")
            .clone();
        (reader, desc)
    }

    /// **F16 decoder** — synthesise a 4-element F16 tensor
    /// with values [1.0, 2.0, -3.0, 0.5], decode, verify
    /// bit-exact round-trip within F16 precision (representable
    /// values).
    #[test]
    fn decode_f16_round_trip() {
        let values = [1.0_f32, 2.0, -3.0, 0.5];
        let mut data: Vec<u8> = Vec::with_capacity(values.len() * 2);
        for &v in &values {
            data.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
        }
        let (reader, desc) = write_minimal_gguf_and_open(
            "tensor.f16",
            GgufTensorType::F16,
            &[values.len() as u64],
            &data,
            "atenia_gguf_decode_f16.gguf",
        );
        let out = decode_tensor(&reader, &desc).expect("decode");
        let _ = std::fs::remove_file(&reader.file_path);
        assert_eq!(out.len(), values.len());
        for (i, &v) in values.iter().enumerate() {
            // Each test value (1.0, 2.0, -3.0, 0.5) is exactly
            // representable in F16 — round-trip is bit-exact.
            assert_eq!(out[i], v, "F16 element {i} mismatch");
        }
    }

    /// **Q8_0 unit (scale = 1.0)** — decoder applies the
    /// formula `y[i] = scale * qs[i]` with no zero-point.
    /// scale=1.0 + qs=[0..32] → [0.0, 1.0, ..., 31.0].
    #[test]
    fn decode_q8_0_identity_scale_one() {
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = i as i8;
        }
        let block = build_q8_0_block(1.0, &qs);
        let (reader, desc) = write_minimal_gguf_and_open(
            "tensor.q8_0",
            GgufTensorType::Q8_0,
            &[32],
            &block,
            "atenia_gguf_decode_q8_identity.gguf",
        );
        let out = decode_tensor(&reader, &desc).expect("decode");
        let _ = std::fs::remove_file(&reader.file_path);
        assert_eq!(out.len(), 32);
        for i in 0..32 {
            // 1.0 and small integers are exactly
            // representable in F16, so the decoded value is
            // bit-exact.
            assert_eq!(out[i], i as f32, "Q8_0 identity element {i} mismatch");
        }
    }

    /// **Q8_0 with non-trivial scale** — scale=2.5 (exactly
    /// representable in F16), qs=[1, -1, 64, -64, ...].
    /// Verifies sign handling on the i8 path and the
    /// scale multiply.
    #[test]
    fn decode_q8_0_with_scale() {
        let qs: [i8; 32] = [
            1, -1, 64, -64, 100, -100, 127, -127, 0, 32, -32, 50, -50, 10, -10, 5, -5, 1, 2, 3, 4,
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ];
        let scale = 2.5_f32;
        let block = build_q8_0_block(scale, &qs);
        let (reader, desc) = write_minimal_gguf_and_open(
            "tensor.q8_0",
            GgufTensorType::Q8_0,
            &[32],
            &block,
            "atenia_gguf_decode_q8_scale.gguf",
        );
        let out = decode_tensor(&reader, &desc).expect("decode");
        let _ = std::fs::remove_file(&reader.file_path);
        assert_eq!(out.len(), 32);
        for i in 0..32 {
            let expected = scale * (qs[i] as f32);
            assert_eq!(
                out[i], expected,
                "Q8_0 scale-aware element {i}: scale={scale}, q={}, expected={expected}",
                qs[i]
            );
            assert!(out[i].is_finite(), "Q8_0 produced non-finite at {i}");
        }
    }

    /// **Q8_0 multi-block** — two blocks back-to-back; the
    /// per-block scale must apply only to its own 32 elements.
    /// Block A scale=1.0, qs=all 1s → 32 ones.
    /// Block B scale=10.0, qs=all 1s → 32 tens.
    #[test]
    fn decode_q8_0_multi_block_per_block_scale() {
        let qs_ones = [1i8; 32];
        let mut data: Vec<u8> = Vec::new();
        data.extend(build_q8_0_block(1.0, &qs_ones));
        data.extend(build_q8_0_block(10.0, &qs_ones));
        let (reader, desc) = write_minimal_gguf_and_open(
            "tensor.q8_0.multi",
            GgufTensorType::Q8_0,
            &[64],
            &data,
            "atenia_gguf_decode_q8_multi.gguf",
        );
        let out = decode_tensor(&reader, &desc).expect("decode");
        let _ = std::fs::remove_file(&reader.file_path);
        assert_eq!(out.len(), 64);
        for i in 0..32 {
            assert_eq!(out[i], 1.0, "block A element {i} should be 1.0");
        }
        for i in 32..64 {
            assert_eq!(out[i], 10.0, "block B element {i} should be 10.0");
        }
    }

    /// **Q8_0 numel-not-multiple-of-32** — a descriptor that
    /// claims 33 elements (one full block + one orphan) is
    /// undefined under the spec; the decoder fails fast with
    /// a clear `InvalidFormat`.
    #[test]
    fn decode_q8_0_rejects_non_multiple_numel() {
        // Build a header that claims 33 elements but supply
        // bytes for one full block (the data byte count
        // doesn't matter for this test — the decoder rejects
        // before reading data).
        let block = build_q8_0_block(1.0, &[0i8; 32]);
        let (reader, desc) = write_minimal_gguf_and_open(
            "tensor.bad",
            GgufTensorType::Q8_0,
            &[33],
            &block,
            "atenia_gguf_decode_q8_bad_numel.gguf",
        );
        let err = decode_tensor(&reader, &desc).expect_err("must fail");
        let _ = std::fs::remove_file(&reader.file_path);
        match err {
            GgufError::InvalidFormat(msg) => {
                assert!(
                    msg.contains("Q8_0") && msg.contains("33") && msg.contains("32"),
                    "error should mention Q8_0, the bad numel, and the block size: {msg}"
                );
            }
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    #[test]
    fn decode_q4_k_synthetic_block() {
        let scales_mins = [
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (7, 6),
            (8, 7),
        ];
        let block = build_q4_k_block(0.5, 0.25, &scales_mins);
        let (reader, desc) = write_minimal_gguf_and_open(
            "tensor.q4_k",
            GgufTensorType::Q4_K,
            &[256],
            &block,
            "atenia_gguf_decode_q4_k.gguf",
        );
        let out = decode_tensor(&reader, &desc).expect("decode");
        let _ = std::fs::remove_file(&reader.file_path);
        assert_eq!(out.len(), 256);
        let mut idx = 0;
        for j in 0..4 {
            let (sc1, min1) = scales_mins[j * 2];
            let (sc2, min2) = scales_mins[j * 2 + 1];
            for l in 0..32 {
                let q = ((l + j) & 0x0F) as f32;
                let expected = 0.5 * (sc1 as f32) * q - 0.25 * (min1 as f32);
                assert_eq!(out[idx], expected, "Q4_K low nibble block {j} lane {l}");
                idx += 1;
            }
            for l in 0..32 {
                let q = (15usize.wrapping_sub(l).wrapping_add(j) & 0x0F) as f32;
                let expected = 0.5 * (sc2 as f32) * q - 0.25 * (min2 as f32);
                assert_eq!(out[idx], expected, "Q4_K high nibble block {j} lane {l}");
                idx += 1;
            }
        }
    }

    /// **Q5_K synthetic block** — same scale/min split as Q4_K
    /// plus the `qh` 5th-bit plane. Re-derives the expected
    /// dequant independently from the canonical llama.cpp
    /// `dequantize_row_q5_K` so a packing / bit-selector bug is
    /// caught (not a tautology against the implementation).
    #[test]
    fn decode_q5_k_synthetic_block() {
        let scales_mins = [
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (7, 6),
            (8, 7),
        ];
        let d = 0.5_f32;
        let dmin = 0.25_f32;
        let block = build_q5_k_block(d, dmin, &scales_mins);
        let (reader, desc) = write_minimal_gguf_and_open(
            "tensor.q5_k",
            GgufTensorType::Q5_K,
            &[256],
            &block,
            "atenia_gguf_decode_q5_k.gguf",
        );
        let out = decode_tensor(&reader, &desc).expect("decode");
        let _ = std::fs::remove_file(&reader.file_path);
        assert_eq!(out.len(), 256);

        // Mirror build_q5_k_block: qh[l] = l, qs[i] = i.
        let mut idx = 0;
        for j in 0..4 {
            let (sc1, min1) = scales_mins[j * 2];
            let (sc2, min2) = scales_mins[j * 2 + 1];
            let d1 = d * (sc1 as f32);
            let m1 = dmin * (min1 as f32);
            let d2 = d * (sc2 as f32);
            let m2 = dmin * (min2 as f32);
            let u1: u8 = 1 << (2 * j);
            let u2: u8 = 2 << (2 * j);
            for l in 0..32 {
                let qs = (j * 32 + l) as u8; // qs[i] = i
                let qh = l as u8; // qh[l] = l
                let hi = if qh & u1 != 0 { 16.0 } else { 0.0 };
                let expected = d1 * (((qs & 0x0F) as f32) + hi) - m1;
                assert_eq!(out[idx], expected, "Q5_K low nibble group {j} lane {l}");
                idx += 1;
            }
            for l in 0..32 {
                let qs = (j * 32 + l) as u8;
                let qh = l as u8;
                let hi = if qh & u2 != 0 { 16.0 } else { 0.0 };
                let expected = d2 * (((qs >> 4) as f32) + hi) - m2;
                assert_eq!(out[idx], expected, "Q5_K high nibble group {j} lane {l}");
                idx += 1;
            }
        }
    }

    /// Q5_K decoder rejects a `numel` that is not a multiple of
    /// the 256-element block (mirrors the Q4_K guard).
    #[test]
    fn decode_q5_k_rejects_non_multiple_numel() {
        let block = build_q5_k_block(1.0, 0.0, &[(1, 0); 8]);
        let (reader, desc) = write_minimal_gguf_and_open(
            "tensor.q5_k.bad",
            GgufTensorType::Q5_K,
            &[255],
            &block,
            "atenia_gguf_decode_q5_k_bad.gguf",
        );
        let err = decode_tensor(&reader, &desc).expect_err("must fail");
        let _ = std::fs::remove_file(&reader.file_path);
        assert!(
            matches!(err, GgufError::InvalidFormat(_)),
            "expected InvalidFormat, got {err:?}"
        );
    }

    #[test]
    fn decode_q4_k_rejects_non_multiple_numel() {
        let block = build_q4_k_block(1.0, 0.0, &[(1, 0); 8]);
        let (reader, desc) = write_minimal_gguf_and_open(
            "tensor.q4_k.bad",
            GgufTensorType::Q4_K,
            &[255],
            &block,
            "atenia_gguf_decode_q4_k_bad_numel.gguf",
        );
        let err = decode_tensor(&reader, &desc).expect_err("must fail");
        let _ = std::fs::remove_file(&reader.file_path);
        match err {
            GgufError::InvalidFormat(msg) => {
                assert!(
                    msg.contains("Q4_K") && msg.contains("255") && msg.contains("256"),
                    "error should mention Q4_K, the bad numel, and the block size: {msg}"
                );
            }
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    /// **Production smoke** — decode the real
    /// `token_embd.weight` of TinyLlama Q8_0 and verify:
    ///   - shape numel matches dimensions
    ///   - every value is finite (no NaN, no Inf)
    ///   - the magnitude is in the embedding range (\|x\| ≤ 1.0
    ///     is typical; we use the looser ≤ 5.0 threshold to
    ///     account for outlier rows)
    ///   - the first element is non-zero (smoke against a
    ///     decoder that silently zeroes its output)
    ///
    /// Skips itself when the local GGUF fixture is absent.
    #[test]
    fn decode_tinyllama_token_embd() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/tinyllama-q8_0/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");
        if !path.exists() {
            eprintln!(
                "[skip] TinyLlama GGUF not found at {}; download via \
                 huggingface_hub `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`",
                path.display()
            );
            return;
        }
        let reader = GgufReader::read_from_path(&path).expect("GGUF parses");
        let embd = reader
            .tensor_by_name("token_embd.weight")
            .expect("token_embd.weight present")
            .clone();
        // GGUF dim order = [in, out] for 2-D linears, but for
        // an embedding table the 2-D shape is [embd_size,
        // vocab_size] in GGUF order. TinyLlama: embd=2048,
        // vocab=32000.
        assert_eq!(embd.dimensions.len(), 2, "embedding rank must be 2");
        assert_eq!(embd.tensor_type, GgufTensorType::Q8_0);
        let numel: u64 = embd.dimensions.iter().product();
        assert_eq!(numel, 32_000 * 2_048, "TinyLlama embed numel mismatch");

        let out = decode_tensor(&reader, &embd).expect("decode succeeds");
        assert_eq!(out.len(), numel as usize, "decoded len matches numel");

        // First element non-zero.
        assert_ne!(
            out[0], 0.0,
            "first embedding element is exactly 0.0 — likely a decoder bug"
        );
        // Sample 10 000 values uniformly to keep the test
        // fast (~tens of ms vs scanning 65 M values).
        let stride = (out.len() / 10_000).max(1);
        let mut max_abs = 0.0_f32;
        for i in (0..out.len()).step_by(stride) {
            let v = out[i];
            assert!(
                v.is_finite(),
                "non-finite at index {i}: {v} — Q8_0 decoder leaks NaN/Inf"
            );
            if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }
        // Llama embedding magnitudes typically ≤ ~0.5; 5.0 is
        // a comfortable upper bound that catches a stray
        // factor-of-1000 byte-order or scale-bit bug.
        assert!(
            max_abs < 5.0,
            "max |embed| = {max_abs} exceeds 5.0 sanity bound — \
             likely a Q8_0 byte-layout or scale bug"
        );

        // Print the diagnostic so the test report carries the
        // production-shape numbers without forcing the
        // operator to instrument by hand.
        eprintln!(
            "[M11.D.2] TinyLlama token_embd.weight decoded: numel={}, \
             out[0]={:.6e}, max|x|={:.4} (sampled stride={})",
            out.len(),
            out[0],
            max_abs,
            stride
        );
    }

    #[test]
    fn decode_tinyllama_q4_k_m_real_tensor() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",
        );
        if !path.exists() {
            eprintln!(
                "[skip] TinyLlama Q4_K_M GGUF not found at {}; download the GGUF fixture first",
                path.display()
            );
            return;
        }
        let reader = GgufReader::read_from_path(&path).expect("GGUF parses");
        let q4_tensor = reader
            .tensors
            .iter()
            .find(|t| t.tensor_type == GgufTensorType::Q4_K)
            .expect("at least one Q4_K tensor present")
            .clone();
        let numel: u64 = q4_tensor.dimensions.iter().product();
        assert_eq!(
            numel as usize % Q4_K_BLOCK_ELEMS,
            0,
            "Q4_K tensor numel must be block-aligned"
        );
        let out = decode_tensor(&reader, &q4_tensor).expect("Q4_K decode succeeds");
        assert_eq!(out.len(), numel as usize);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(
            out.iter().any(|v| *v != 0.0),
            "decoded Q4_K tensor should not be all zeros"
        );
    }
}
