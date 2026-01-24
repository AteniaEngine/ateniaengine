#![allow(dead_code)]

/// Simple deterministic FNV-1a 64-bit hash encoded as lowercase hex.
pub fn hash_bytes(data: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

pub fn hash_str(s: &str) -> String {
    hash_bytes(s.as_bytes())
}
