pub fn chunk_size(tensor_len: usize) -> usize {
    if tensor_len < 2048 {
        tensor_len
    } else {
        tensor_len / 8
    }
}
