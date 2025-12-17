use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::tensor::{Device, DType, Layout, Tensor};

#[derive(Clone, Debug)]
pub struct Vocab {
    pub stoi: HashMap<char, usize>,
    pub itos: Vec<char>,
}

impl Vocab {
    pub fn size(&self) -> usize {
        self.itos.len()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&idx| self.itos.get(idx).copied().unwrap_or('?'))
            .collect()
    }
}

pub fn load_text_dataset(path: &str) -> std::io::Result<String> {
    fs::read_to_string(Path::new(path))
}

pub fn build_vocab(text: &str) -> Vocab {
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort();
    chars.dedup();

    let itos = chars;
    let stoi = itos
        .iter()
        .enumerate()
        .map(|(idx, ch)| (*ch, idx))
        .collect::<HashMap<_, _>>();

    Vocab { stoi, itos }
}

pub fn encode_text(text: &str, vocab: &Vocab) -> Vec<usize> {
    text.chars()
        .map(|ch| *vocab.stoi.get(&ch).expect("character missing from vocabulary"))
        .collect()
}

/// Create autoregressive batches (inputs, targets) with shape [batch_size, seq_len].
pub fn make_batches(tokens: Vec<usize>, seq_len: usize, batch_size: usize) -> Vec<(Tensor, Tensor)> {
    assert!(seq_len > 0, "seq_len must be positive");
    assert!(batch_size > 0, "batch_size must be positive");

    let window = seq_len + 1;
    let total_windows = tokens.len() / window;
    let batches = total_windows / batch_size;
    let usable_tokens = batches * batch_size * window;

    let mut batches_out = Vec::with_capacity(batches);
    for b in 0..batches {
        let mut input = Tensor::with_layout(
            vec![batch_size, seq_len],
            0.0,
            Device::CPU,
            Layout::Contiguous,
            DType::F32,
        );
        let mut target = Tensor::with_layout(
            vec![batch_size, seq_len],
            0.0,
            Device::CPU,
            Layout::Contiguous,
            DType::F32,
        );

        for row in 0..batch_size {
            let window_idx = b * batch_size + row;
            let start = window_idx * window;
            if start + window > usable_tokens {
                break;
            }
            for offset in 0..seq_len {
                let x = tokens[start + offset] as f32;
                let y = tokens[start + offset + 1] as f32;
                let dst = row * seq_len + offset;
                input.data[dst] = x;
                target.data[dst] = y;
            }
        }

        batches_out.push((input, target));
    }

    batches_out
}
