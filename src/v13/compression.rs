use super::memory_types::{CompressionKind, CompressionMeta, MoveError};

pub fn rle_compress(input: &[u8]) -> (Vec<u8>, CompressionMeta) {
    let mut out = Vec::new();
    if input.is_empty() {
        return (
            out,
            CompressionMeta {
                kind: CompressionKind::None,
                original_bytes: 0,
            },
        );
    }

    let mut i = 0usize;
    while i < input.len() {
        let value = input[i];
        let mut run_len: u8 = 1;
        i += 1;

        while i < input.len() && input[i] == value && run_len < u8::MAX {
            run_len = run_len.saturating_add(1);
            i += 1;
        }

        out.push(run_len);
        out.push(value);
    }

    (
        out,
        CompressionMeta {
            kind: CompressionKind::Rle,
            original_bytes: input.len() as u64,
        },
    )
}

pub fn rle_decompress(input: &[u8], meta: &CompressionMeta) -> Result<Vec<u8>, MoveError> {
    if input.len() % 2 != 0 {
        return Err(MoveError::Unsupported(
            "Invalid RLE stream: odd length".to_string(),
        ));
    }

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < input.len() {
        let count = input[i];
        let value = input[i + 1];
        i += 2;

        if count == 0 {
            return Err(MoveError::Unsupported(
                "Invalid RLE stream: zero count".to_string(),
            ));
        }

        out.extend(std::iter::repeat(value).take(count as usize));
    }

    if out.len() as u64 != meta.original_bytes {
        return Err(MoveError::Unsupported(
            "Invalid RLE stream: length mismatch".to_string(),
        ));
    }

    Ok(out)
}
