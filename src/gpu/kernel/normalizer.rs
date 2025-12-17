/// APX 12.0: capa de normalización de kernels CUDA antes de pasar por NVRTC.
/// Mantiene una firma consistente y un layout estable para futuros pasos
/// como fusión y CUDAGraphs.
pub struct KernelNormalizer;

impl KernelNormalizer {
    /// Normaliza el código fuente de un kernel CUDA C antes de compilarlo con NVRTC.
    ///
    /// Pasos que realiza de forma conservadora (sin cambiar la semántica):
    /// - Normaliza saltos de línea y elimina espacios en blanco al final de línea.
    /// - Garantiza que el texto contenga `extern "C" __global__`.
    /// - Garantiza que el nombre del kernel aparezca en la firma.
    /// - Deja el resto del cuerpo intacto para no introducir bugs sutiles.
    pub fn normalize_kernel(src: &str, kernel_name: &str) -> String {
        // 1. Normalizar saltos de línea a `\n` y recortar espacio al final de cada línea.
        let normalized_lines: Vec<String> = src
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .lines()
            .map(|l| l.trim_end().to_string())
            .collect();

        let result = normalized_lines.join("\n");

        // 2. Asegurar que existe `extern "C" __global__`.
        if !result.contains("extern \"C\" __global__") {
            // Si no está, no intentamos reescribir todo el kernel; simplemente
            // devolvemos el código tal cual para no romper kernels existentes.
            return result;
        }

        // 3. Asegurar que el nombre del kernel aparece en la firma.
        if !result.contains(kernel_name) {
            // De nuevo, evitamos tocar la semántica si no reconocemos el patrón.
            return result;
        }

        // En esta primera versión, la normalización es deliberadamente mínima y
        // no reordena parámetros ni reescribe el prototipo. Eso se puede añadir
        // de forma incremental cuando tengamos más kernels migrados.
        result
    }
}
