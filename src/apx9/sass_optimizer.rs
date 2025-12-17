// APX 9.5 — SASS Optimizer Mock v0
// Optimización sintética de SASS basada únicamente en manipulación de texto.

pub struct SassOptimizer;

impl SassOptimizer {
    pub fn optimize(sass: &str) -> String {
        if sass.trim().is_empty() || sass.contains("invalid") {
            return sass.into();
        }

        let mut out = String::new();
        out.push_str("// SASS OPT v0\n");

        let mut lines: Vec<_> = sass.lines().collect();

        // 1. Eliminar NOPs sintéticos
        lines.retain(|l| !l.trim().starts_with("NOP"));

        // 2. Reordenar: LDG primero, luego FADD, luego STG
        lines.sort_by_key(|l| {
            let l = l.trim();
            if l.starts_with("LDG") { 0 }
            else if l.starts_with("FADD") { 1 }
            else if l.starts_with("STG") { 2 }
            else { 3 }
        });

        // 3. Normalizar espacios
        for l in lines {
            let trimmed = l.trim();
            if trimmed.is_empty() { continue; }
            out.push_str(trimmed);
            out.push('\n');
        }

        out
    }
}
