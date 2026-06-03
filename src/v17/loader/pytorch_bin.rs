//! **FORMAT-INTAKE-1** — PyTorch `.bin` (`torch.save`) → safetensors transcoder.
//!
//! A `pytorch_model.bin` produced by `torch.save(state_dict)` is a **ZIP
//! archive** (entries stored uncompressed) containing:
//!   - `<root>/data.pkl` — a Python **pickle** stream describing the state dict
//!     (an `OrderedDict[str, Tensor]`); each tensor is built by
//!     `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, …)` where
//!     `storage` is a `persistent_id` reference to a separate zip entry;
//!   - `<root>/data/<key>` — the raw little-endian tensor storage bytes;
//!   - `<root>/byteorder`, `<root>/version` — metadata.
//!
//! Rather than teach the whole load pipeline a new on-disk format, this module
//! **transcodes** a `.bin` into an in-memory **safetensors** byte buffer, which
//! the existing [`super::safetensors_reader::SafetensorsReader::from_bytes`]
//! consumes unchanged — so the weight mapper, adapter layer, transforms and
//! tier planning are all reused with **zero** modification.
//!
//! ## Safety
//!
//! Pickle can encode arbitrary Python object construction (a remote-code-
//! execution surface for naïve unpicklers). This module ships a **restricted**
//! unpickler: it implements only the opcode subset `torch.save` emits for a
//! state dict, and it accepts only a **whitelist** of globals
//! (`collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` /
//! `_rebuild_parameter`, and the `torch.*Storage` dtype tags). **Any** other
//! global, opcode, or structure is a hard error — never executed, never
//! silently skipped.
//!
//! ## Supported / rejected (fail-loud, no silent fallback)
//!
//! - **Supported:** modern `torch.save` ZIP format (pickle protocol 2–5);
//!   flat `OrderedDict[str, Tensor]`; contiguous tensors (`offset == 0`,
//!   default stride); dtypes Float (→F32), Half (→F16), BFloat16 (→BF16);
//!   little-endian storages.
//! - **Rejected with a clear error:** legacy non-zip pickle (`torch<1.6`);
//!   compressed zip entries; non-contiguous / storage-sharing views;
//!   Double/Long/Int/Bool/… storages; big-endian; ZIP64; unknown globals or
//!   opcodes; nested/unexpected pickle structure.

use std::collections::BTreeMap;
use std::path::Path;

use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

use super::loader_errors::LoaderError;

fn err(msg: impl Into<String>) -> LoaderError {
    LoaderError::InvalidFormat(msg.into())
}

// ===========================================================================
// Minimal ZIP reader (STORED entries only — torch.save never compresses)
// ===========================================================================

struct Zip<'a> {
    bytes: &'a [u8],
    /// entry name → (data_start, len) for STORED entries.
    entries: BTreeMap<String, (usize, usize)>,
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl<'a> Zip<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, LoaderError> {
        // Locate the End Of Central Directory record (signature 0x06054b50),
        // scanning backwards (the EOCD comment is virtually always empty).
        const EOCD_SIG: u32 = 0x0605_4b50;
        if bytes.len() < 22 {
            return Err(err("pytorch .bin: file too small to be a zip archive"));
        }
        let mut eocd = None;
        let scan_start = bytes.len().saturating_sub(22 + 0xffff);
        for i in (scan_start..=bytes.len() - 22).rev() {
            if u32le(bytes, i) == EOCD_SIG {
                eocd = Some(i);
                break;
            }
        }
        let eocd = eocd.ok_or_else(|| {
            err("pytorch .bin: not a zip archive (no End-Of-Central-Directory record). \
                 Legacy torch<1.6 non-zip checkpoints are not supported — re-save with a \
                 modern torch or convert to safetensors.")
        })?;
        // ZIP64 sentinel values → unsupported (very large checkpoints).
        let total_entries = u16le(bytes, eocd + 10);
        let cd_offset = u32le(bytes, eocd + 16) as usize;
        if total_entries == 0xffff || cd_offset as u32 == 0xffff_ffff {
            return Err(err(
                "pytorch .bin: ZIP64 archive not supported (checkpoint ≥ 4 GiB / ≥ 65535 \
                 entries) — convert to safetensors.",
            ));
        }

        let mut entries = BTreeMap::new();
        let mut p = cd_offset;
        const CDH_SIG: u32 = 0x0201_4b50;
        for _ in 0..total_entries {
            if p + 46 > bytes.len() || u32le(bytes, p) != CDH_SIG {
                return Err(err("pytorch .bin: malformed zip central directory"));
            }
            let method = u16le(bytes, p + 10);
            let comp_size = u32le(bytes, p + 20) as usize;
            let uncomp_size = u32le(bytes, p + 24) as usize;
            let name_len = u16le(bytes, p + 28) as usize;
            let extra_len = u16le(bytes, p + 30) as usize;
            let comment_len = u16le(bytes, p + 32) as usize;
            let local_off = u32le(bytes, p + 42) as usize;
            let name = std::str::from_utf8(&bytes[p + 46..p + 46 + name_len])
                .map_err(|_| err("pytorch .bin: non-UTF8 zip entry name"))?
                .to_string();
            if method != 0 {
                return Err(err(format!(
                    "pytorch .bin: zip entry {name:?} is compressed (method {method}); \
                     torch.save uses STORED — convert to safetensors."
                )));
            }
            if comp_size != uncomp_size {
                return Err(err(format!(
                    "pytorch .bin: zip entry {name:?} comp/uncomp size mismatch on a STORED entry"
                )));
            }
            // Resolve the data start via the LOCAL header (its extra-field
            // length may differ from the central directory's).
            const LFH_SIG: u32 = 0x0403_4b50;
            if local_off + 30 > bytes.len() || u32le(bytes, local_off) != LFH_SIG {
                return Err(err(format!(
                    "pytorch .bin: bad local header for zip entry {name:?}"
                )));
            }
            let l_name_len = u16le(bytes, local_off + 26) as usize;
            let l_extra_len = u16le(bytes, local_off + 28) as usize;
            let data_start = local_off + 30 + l_name_len + l_extra_len;
            if data_start + uncomp_size > bytes.len() {
                return Err(err(format!(
                    "pytorch .bin: zip entry {name:?} data range out of bounds"
                )));
            }
            entries.insert(name, (data_start, uncomp_size));
            p += 46 + name_len + extra_len + comment_len;
        }
        Ok(Zip { bytes, entries })
    }

    fn read(&self, name: &str) -> Option<&'a [u8]> {
        let &(start, len) = self.entries.get(name)?;
        Some(&self.bytes[start..start + len])
    }

    /// The archive root directory, e.g. `"pytorch_model"` for the entry
    /// `"pytorch_model/data.pkl"`.
    fn find_pkl_root(&self) -> Result<String, LoaderError> {
        let pkl = self
            .entries
            .keys()
            .find(|k| k.ends_with("/data.pkl") || k.as_str() == "data.pkl")
            .ok_or_else(|| err("pytorch .bin: no data.pkl entry in the archive"))?;
        Ok(pkl
            .strip_suffix("data.pkl")
            .unwrap()
            .trim_end_matches('/')
            .to_string())
    }
}

// ===========================================================================
// Restricted pickle VM
// ===========================================================================

#[derive(Clone, Debug)]
enum Val {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Tuple(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global(String),
    Storage(StorageRef),
    Tensor(TensorSpec),
    Mark,
}

#[derive(Clone, Debug)]
struct StorageRef {
    dtype: StDtype,
    key: String,
    numel: usize,
}

#[derive(Clone, Debug)]
struct TensorSpec {
    storage: StorageRef,
    offset: i64,
    size: Vec<usize>,
    stride: Vec<usize>,
}

fn storage_dtype(global: &str) -> Result<StDtype, LoaderError> {
    Ok(match global {
        "torch.FloatStorage" => StDtype::F32,
        "torch.HalfStorage" => StDtype::F16,
        "torch.BFloat16Storage" => StDtype::BF16,
        other => {
            return Err(err(format!(
                "pytorch .bin: storage type {other} is not supported (only Float/Half/BFloat16). \
                 Double/Long/Int/Bool tensors cannot be loaded — convert to safetensors."
            )))
        }
    })
}

struct Unpickler<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Val>,
    memo: Vec<Val>,
}

impl<'a> Unpickler<'a> {
    fn new(data: &'a [u8]) -> Self {
        Unpickler {
            data,
            pos: 0,
            stack: Vec::new(),
            memo: Vec::new(),
        }
    }

    fn byte(&mut self) -> Result<u8, LoaderError> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| err("pytorch .bin: pickle stream truncated"))?;
        self.pos += 1;
        Ok(b)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], LoaderError> {
        if self.pos + n > self.data.len() {
            return Err(err("pytorch .bin: pickle stream truncated"));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn line(&mut self) -> Result<String, LoaderError> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.data[start..self.pos])
            .map_err(|_| err("pytorch .bin: non-UTF8 pickle line"))?
            .to_string();
        self.pos += 1; // skip '\n'
        Ok(s)
    }
    fn pop(&mut self) -> Result<Val, LoaderError> {
        self.stack
            .pop()
            .ok_or_else(|| err("pytorch .bin: pickle stack underflow"))
    }
    fn pop_to_mark(&mut self) -> Result<Vec<Val>, LoaderError> {
        let mut items = Vec::new();
        loop {
            match self.stack.pop() {
                Some(Val::Mark) => break,
                Some(v) => items.push(v),
                None => return Err(err("pytorch .bin: pickle MARK not found")),
            }
        }
        items.reverse();
        Ok(items)
    }
    fn memo_put(&mut self, idx: usize) -> Result<(), LoaderError> {
        let top = self
            .stack
            .last()
            .ok_or_else(|| err("pytorch .bin: pickle memo put on empty stack"))?
            .clone();
        if idx >= self.memo.len() {
            self.memo.resize(idx + 1, Val::None);
        }
        self.memo[idx] = top;
        Ok(())
    }
    fn memo_get(&mut self, idx: usize) -> Result<(), LoaderError> {
        let v = self
            .memo
            .get(idx)
            .cloned()
            .ok_or_else(|| err("pytorch .bin: pickle memo get out of range"))?;
        self.stack.push(v);
        Ok(())
    }

    fn reduce(&mut self) -> Result<(), LoaderError> {
        let args = self.pop()?;
        let func = self.pop()?;
        let args = match args {
            Val::Tuple(a) => a,
            other => return Err(err(format!("pytorch .bin: REDUCE args not a tuple: {other:?}"))),
        };
        let name = match &func {
            Val::Global(g) => g.as_str(),
            other => return Err(err(format!("pytorch .bin: REDUCE callable not a global: {other:?}"))),
        };
        match name {
            "collections.OrderedDict" => self.stack.push(Val::Dict(Vec::new())),
            "torch._utils._rebuild_tensor_v2" => {
                // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, ...])
                if args.len() < 4 {
                    return Err(err("pytorch .bin: _rebuild_tensor_v2 needs ≥4 args"));
                }
                let storage = match &args[0] {
                    Val::Storage(s) => s.clone(),
                    other => return Err(err(format!("pytorch .bin: tensor storage not a persistent storage: {other:?}"))),
                };
                let offset = match &args[1] {
                    Val::Int(i) => *i,
                    other => return Err(err(format!("pytorch .bin: tensor offset not int: {other:?}"))),
                };
                let size = as_usize_tuple(&args[2], "size")?;
                let stride = as_usize_tuple(&args[3], "stride")?;
                self.stack.push(Val::Tensor(TensorSpec {
                    storage,
                    offset,
                    size,
                    stride,
                }));
            }
            "torch._utils._rebuild_parameter" => {
                // (tensor, requires_grad, backward_hooks) → unwrap to the tensor.
                let inner = args
                    .into_iter()
                    .next()
                    .ok_or_else(|| err("pytorch .bin: _rebuild_parameter missing tensor"))?;
                match inner {
                    Val::Tensor(_) => self.stack.push(inner),
                    other => return Err(err(format!("pytorch .bin: _rebuild_parameter inner not a tensor: {other:?}"))),
                }
            }
            other => {
                return Err(err(format!(
                    "pytorch .bin: refusing to reduce unknown global {other:?} \
                     (restricted unpickler — only OrderedDict / _rebuild_tensor_v2 / \
                     _rebuild_parameter are allowed)"
                )))
            }
        }
        Ok(())
    }

    fn persistent_id(&mut self) -> Result<(), LoaderError> {
        // pid tuple: ('storage', <storage_type_global>, key, location, numel)
        let pid = self.pop()?;
        let t = match pid {
            Val::Tuple(t) => t,
            other => return Err(err(format!("pytorch .bin: persistent_id not a tuple: {other:?}"))),
        };
        if t.len() != 5 {
            return Err(err(format!(
                "pytorch .bin: unexpected persistent_id arity {} (expected 5)",
                t.len()
            )));
        }
        match (&t[0], &t[1], &t[2], &t[4]) {
            (Val::Str(tag), Val::Global(stype), Val::Str(key), Val::Int(numel)) if tag == "storage" => {
                self.stack.push(Val::Storage(StorageRef {
                    dtype: storage_dtype(stype)?,
                    key: key.clone(),
                    numel: *numel as usize,
                }));
                Ok(())
            }
            _ => Err(err("pytorch .bin: malformed storage persistent_id")),
        }
    }

    /// Run to STOP, returning the final `OrderedDict[str, Tensor]` as pairs.
    fn run(mut self) -> Result<Vec<(String, TensorSpec)>, LoaderError> {
        loop {
            let op = self.byte()?;
            match op {
                0x80 => {
                    self.byte()?; // PROTO version
                }
                0x95 => {
                    self.take(8)?; // FRAME length
                }
                b'(' => self.stack.push(Val::Mark),
                b')' => self.stack.push(Val::Tuple(Vec::new())),
                b'}' => self.stack.push(Val::Dict(Vec::new())),
                b']' => self.stack.push(Val::Tuple(Vec::new())), // EMPTY_LIST (unused; treat as tuple)
                b'N' => self.stack.push(Val::None),
                0x88 => self.stack.push(Val::Bool(true)),
                0x89 => self.stack.push(Val::Bool(false)),
                b't' => {
                    let items = self.pop_to_mark()?;
                    self.stack.push(Val::Tuple(items));
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a, b, c]));
                }
                b'K' => {
                    let v = self.byte()? as i64;
                    self.stack.push(Val::Int(v));
                }
                b'M' => {
                    let b = self.take(2)?;
                    self.stack.push(Val::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
                }
                b'J' => {
                    let b = self.take(4)?;
                    self.stack.push(Val::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64));
                }
                0x8a => {
                    // LONG1: 1-byte length then little-endian signed.
                    let n = self.byte()? as usize;
                    let b = self.take(n)?;
                    let mut val: i64 = 0;
                    for (i, &byte) in b.iter().enumerate() {
                        val |= (byte as i64) << (8 * i);
                    }
                    if n > 0 && b[n - 1] & 0x80 != 0 && n < 8 {
                        val |= -1i64 << (8 * n); // sign-extend
                    }
                    self.stack.push(Val::Int(val));
                }
                b'X' => {
                    let l = self.take(4)?;
                    let len = u32::from_le_bytes([l[0], l[1], l[2], l[3]]) as usize;
                    let s = self.take(len)?;
                    self.stack.push(Val::Str(
                        std::str::from_utf8(s)
                            .map_err(|_| err("pytorch .bin: BINUNICODE not UTF8"))?
                            .to_string(),
                    ));
                }
                0x8c => {
                    let len = self.byte()? as usize;
                    let s = self.take(len)?;
                    self.stack.push(Val::Str(
                        std::str::from_utf8(s)
                            .map_err(|_| err("pytorch .bin: SHORT_BINUNICODE not UTF8"))?
                            .to_string(),
                    ));
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Val::Global(format!("{module}.{name}")));
                }
                0x93 => {
                    // STACK_GLOBAL: pop name, module (both Str)
                    let name = self.pop()?;
                    let module = self.pop()?;
                    match (module, name) {
                        (Val::Str(m), Val::Str(n)) => self.stack.push(Val::Global(format!("{m}.{n}"))),
                        _ => return Err(err("pytorch .bin: STACK_GLOBAL operands not strings")),
                    }
                }
                b'q' => {
                    let i = self.byte()? as usize;
                    self.memo_put(i)?;
                }
                b'r' => {
                    let b = self.take(4)?;
                    let i = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
                    self.memo_put(i)?;
                }
                0x94 => {
                    let i = self.memo.len();
                    self.memo_put(i)?;
                }
                b'h' => {
                    let i = self.byte()? as usize;
                    self.memo_get(i)?;
                }
                b'j' => {
                    let b = self.take(4)?;
                    let i = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
                    self.memo_get(i)?;
                }
                b'R' => self.reduce()?,
                b'Q' => self.persistent_id()?,
                b's' => {
                    // SETITEM: dict[key] = value
                    let value = self.pop()?;
                    let key = self.pop()?;
                    self.set_items(vec![key, value])?;
                }
                b'u' => {
                    // SETITEMS: pop pairs back to MARK
                    let items = self.pop_to_mark()?;
                    self.set_items(items)?;
                }
                b'.' => break,
                other => {
                    return Err(err(format!(
                        "pytorch .bin: unsupported pickle opcode 0x{other:02x} at offset {} \
                         (restricted unpickler)",
                        self.pos - 1
                    )))
                }
            }
        }

        // Final object is the state dict.
        let top = self.pop()?;
        let pairs = match top {
            Val::Dict(d) => d,
            other => return Err(err(format!("pytorch .bin: top-level object is not a dict: {other:?}"))),
        };
        let mut out = Vec::with_capacity(pairs.len());
        for (k, v) in pairs {
            let name = match k {
                Val::Str(s) => s,
                other => return Err(err(format!("pytorch .bin: state-dict key not a string: {other:?}"))),
            };
            let spec = match v {
                Val::Tensor(t) => t,
                other => {
                    return Err(err(format!(
                        "pytorch .bin: state-dict entry {name:?} is not a tensor: {other:?} \
                         (only flat OrderedDict[str, Tensor] is supported)"
                    )))
                }
            };
            out.push((name, spec));
        }
        Ok(out)
    }

    /// Append `items` (flat key,value,key,value,…) into the dict just below
    /// the MARK position — i.e. the current top of stack.
    fn set_items(&mut self, items: Vec<Val>) -> Result<(), LoaderError> {
        if items.len() % 2 != 0 {
            return Err(err("pytorch .bin: SETITEMS odd number of items"));
        }
        let dict = self
            .stack
            .last_mut()
            .ok_or_else(|| err("pytorch .bin: SETITEMS with empty stack"))?;
        let d = match dict {
            Val::Dict(d) => d,
            other => return Err(err(format!("pytorch .bin: SETITEMS target not a dict: {other:?}"))),
        };
        let mut it = items.into_iter();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            d.push((k, v));
        }
        Ok(())
    }
}

fn as_usize_tuple(v: &Val, what: &str) -> Result<Vec<usize>, LoaderError> {
    match v {
        Val::Tuple(t) => t
            .iter()
            .map(|x| match x {
                Val::Int(i) if *i >= 0 => Ok(*i as usize),
                other => Err(err(format!("pytorch .bin: {what} element not a non-negative int: {other:?}"))),
            })
            .collect(),
        other => Err(err(format!("pytorch .bin: {what} not a tuple: {other:?}"))),
    }
}

fn contiguous_stride(size: &[usize]) -> Vec<usize> {
    let mut stride = vec![1usize; size.len()];
    for i in (0..size.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * size[i + 1];
    }
    stride
}

fn dtype_size(d: StDtype) -> usize {
    match d {
        StDtype::F32 => 4,
        StDtype::F16 | StDtype::BF16 => 2,
        _ => 0,
    }
}

// ===========================================================================
// Public entry points
// ===========================================================================

/// One materialised tensor extracted from a `.bin`: name, safetensors dtype,
/// shape, and the contiguous little-endian bytes.
pub(crate) struct BinTensor {
    pub name: String,
    pub dtype: StDtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Parse one `torch.save` `.bin` buffer into its materialised tensors. Fails
/// loud on any structure outside the supported subset (see module docs).
pub(crate) fn bin_to_tensors(bin_bytes: &[u8]) -> Result<Vec<BinTensor>, LoaderError> {
    let zip = Zip::parse(bin_bytes)?;
    let root = zip.find_pkl_root()?;

    // Byte-order guard (default little when absent).
    let byteorder_entry = if root.is_empty() {
        "byteorder".to_string()
    } else {
        format!("{root}/byteorder")
    };
    if let Some(bo) = zip.read(&byteorder_entry) {
        let bo = std::str::from_utf8(bo).unwrap_or("").trim();
        if bo == "big" {
            return Err(err(
                "pytorch .bin: big-endian storages are not supported — convert to safetensors.",
            ));
        }
    }

    let pkl_entry = if root.is_empty() {
        "data.pkl".to_string()
    } else {
        format!("{root}/data.pkl")
    };
    let pkl = zip
        .read(&pkl_entry)
        .ok_or_else(|| err("pytorch .bin: data.pkl entry missing"))?;
    let tensors = Unpickler::new(pkl).run()?;
    if tensors.is_empty() {
        return Err(err("pytorch .bin: state dict contains no tensors"));
    }

    let mut out: Vec<BinTensor> = Vec::with_capacity(tensors.len());
    for (name, spec) in tensors {
        let dsize = dtype_size(spec.storage.dtype);
        let numel: usize = spec.size.iter().product();
        if spec.offset != 0 {
            return Err(err(format!(
                "pytorch .bin: tensor {name:?} has non-zero storage offset ({}) — \
                 storage-sharing views are not supported; convert to safetensors.",
                spec.offset
            )));
        }
        if spec.stride != contiguous_stride(&spec.size) {
            return Err(err(format!(
                "pytorch .bin: tensor {name:?} is non-contiguous (stride {:?} for size {:?}) — \
                 not supported; convert to safetensors.",
                spec.stride, spec.size
            )));
        }
        if numel != spec.storage.numel {
            return Err(err(format!(
                "pytorch .bin: tensor {name:?} numel {numel} != storage numel {} — \
                 storage-sharing views are not supported.",
                spec.storage.numel
            )));
        }
        let data_entry = if root.is_empty() {
            format!("data/{}", spec.storage.key)
        } else {
            format!("{root}/data/{}", spec.storage.key)
        };
        let storage_bytes = zip.read(&data_entry).ok_or_else(|| {
            err(format!(
                "pytorch .bin: storage entry {data_entry:?} for tensor {name:?} not found"
            ))
        })?;
        let need = numel * dsize;
        if storage_bytes.len() < need {
            return Err(err(format!(
                "pytorch .bin: tensor {name:?} needs {need} bytes but storage has {}",
                storage_bytes.len()
            )));
        }
        out.push(BinTensor {
            name,
            dtype: spec.storage.dtype,
            shape: spec.size,
            bytes: storage_bytes[..need].to_vec(),
        });
    }
    Ok(out)
}

/// Serialize materialised tensors into a safetensors byte buffer.
fn serialize_tensors(tensors: &[BinTensor]) -> Result<Vec<u8>, LoaderError> {
    let mut views: BTreeMap<String, TensorView> = BTreeMap::new();
    for t in tensors {
        let view = TensorView::new(t.dtype, t.shape.clone(), &t.bytes).map_err(|e| {
            err(format!(
                "pytorch .bin: building safetensors view for {:?}: {e:?}",
                t.name
            ))
        })?;
        views.insert(t.name.clone(), view);
    }
    safetensors::serialize(&views, &None)
        .map_err(|e| err(format!("pytorch .bin: safetensors serialize failed: {e:?}")))
}

/// **FORMAT-INTAKE-1** — transcode a single-file `torch.save` `.bin` byte
/// buffer into an equivalent **safetensors** byte buffer (consumable by
/// `SafetensorsReader::from_bytes`).
pub fn transcode_bin_to_safetensors(bin_bytes: &[u8]) -> Result<Vec<u8>, LoaderError> {
    serialize_tensors(&bin_to_tensors(bin_bytes)?)
}

/// **FORMAT-INTAKE-2** — transcode a **sharded** PyTorch checkpoint
/// (`pytorch_model.bin.index.json` + `pytorch_model-0000k-of-000NN.bin`) into a
/// single in-memory safetensors buffer.
///
/// Reuses the FORMAT-INTAKE-1 per-shard transcode and the existing
/// [`super::shard_index::ShardIndex`] parser (same JSON schema as the
/// safetensors index). Every shard is transcoded and its tensors assembled into
/// one logical reader. **Fail-loud consistency** (no silent fallback):
/// - missing / unreadable shard file → error;
/// - the same tensor name in more than one shard → error;
/// - a tensor declared in `weight_map` but absent from the shards → error;
/// - a tensor present in the shards but not declared in `weight_map` → error;
/// - malformed / empty `weight_map`, duplicate index keys → error (via `ShardIndex`).
pub fn transcode_sharded_bin_to_safetensors(index_path: &Path) -> Result<Vec<u8>, LoaderError> {
    use super::shard_index::ShardIndex;
    use std::collections::BTreeSet;

    let index = ShardIndex::from_file(index_path)?;

    let mut all: BTreeMap<String, BinTensor> = BTreeMap::new();
    for shard in index.shard_filenames() {
        let path = index.shard_path(&shard);
        let bytes = std::fs::read(&path).map_err(|e| {
            err(format!(
                "sharded .bin: cannot read shard {} referenced by the index: {e}",
                path.display()
            ))
        })?;
        let tensors = bin_to_tensors(&bytes)
            .map_err(|e| err(format!("sharded .bin: in shard {shard}: {e}")))?;
        for t in tensors {
            let name = t.name.clone();
            if all.insert(name.clone(), t).is_some() {
                return Err(err(format!(
                    "sharded .bin: tensor `{name}` appears in more than one shard"
                )));
            }
        }
    }

    // Cross-check the assembled tensor set against the index's weight_map.
    let declared: BTreeSet<&String> = index.weight_map.keys().collect();
    let assembled: BTreeSet<&String> = all.keys().collect();
    let missing: Vec<&String> = declared.difference(&assembled).map(|s| *s).collect();
    if !missing.is_empty() {
        return Err(err(format!(
            "sharded .bin: weight_map declares tensors absent from the shards: {missing:?}"
        )));
    }
    let extra: Vec<&String> = assembled.difference(&declared).map(|s| *s).collect();
    if !extra.is_empty() {
        return Err(err(format!(
            "sharded .bin: shards contain tensors not declared in weight_map: {extra:?}"
        )));
    }

    let tensors: Vec<BinTensor> = all.into_values().collect();
    serialize_tensors(&tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_zip() {
        let e = transcode_bin_to_safetensors(b"not a zip at all, definitely").unwrap_err();
        assert!(format!("{e}").contains("zip"));
    }

    #[test]
    fn rejects_empty() {
        assert!(transcode_bin_to_safetensors(&[]).is_err());
    }

    #[test]
    fn contiguous_stride_is_row_major() {
        assert_eq!(contiguous_stride(&[2, 3]), vec![3, 1]);
        assert_eq!(contiguous_stride(&[4]), vec![1]);
        assert_eq!(contiguous_stride(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn storage_dtype_whitelist() {
        assert!(storage_dtype("torch.FloatStorage").is_ok());
        assert!(storage_dtype("torch.HalfStorage").is_ok());
        assert!(storage_dtype("torch.BFloat16Storage").is_ok());
        assert!(storage_dtype("torch.DoubleStorage").is_err());
        assert!(storage_dtype("torch.LongStorage").is_err());
    }
}
