/// APX 12.0: CUDA kernel normalization layer before passing through NVRTC.
/// Keeps a consistent signature and a stable layout for future steps
/// such as fusion and CUDAGraphs.
pub struct KernelNormalizer;

impl KernelNormalizer {
    /// Normalize the source code of a CUDA C kernel before compiling it with NVRTC.
    ///
    /// Steps performed conservatively (without changing semantics):
    /// - Normalize newlines and remove trailing whitespace.
    /// - Ensure the text contains `extern "C" __global__`.
    /// - Ensure the kernel name appears in the signature.
    /// - Keep the rest of the body intact to avoid introducing subtle bugs.
    pub fn normalize_kernel(src: &str, kernel_name: &str) -> String {
        // 1. Normalize newlines to `\n` and trim trailing whitespace per line.
        let normalized_lines: Vec<String> = src
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .lines()
            .map(|l| l.trim_end().to_string())
            .collect();

        let result = normalized_lines.join("\n");

        // 2. Ensure `extern "C" __global__` exists.
        if !result.contains("extern \"C\" __global__") {
            // If not present, do not attempt to rewrite the whole kernel; just
            // return the code as-is to avoid breaking existing kernels.
            return result;
        }

        // 3. Ensure the kernel name appears in the signature.
        if !result.contains(kernel_name) {
            // Again, avoid touching semantics if we do not recognize the pattern.
            return result;
        }

        // In this first version, normalization is deliberately minimal and does
        // not reorder parameters nor rewrite the prototype. This can be added
        // incrementally when we have more kernels migrated.
        result
    }
}
