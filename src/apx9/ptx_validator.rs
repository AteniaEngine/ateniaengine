// APX 9.3 — PTX Validator v0
// Validador sintáctico/estructural de PTX puramente textual.
// No ejecuta ni compila PTX, no requiere CUDA ni toolchains externos.

pub struct PtxValidationResult {
    pub ok: bool,
    pub errors: Vec<String>,
}

pub struct PtxValidator;

impl PtxValidator {
    pub fn validate(ptx: &str) -> PtxValidationResult {
        let mut errors = Vec::new();

        // 1. Debe contener .version
        if !ptx.contains(".version") {
            errors.push("Missing .version directive".into());
        }

        // 2. Debe contener .entry
        if !ptx.contains(".entry") {
            errors.push("Missing .entry function".into());
        }

        // 3. Verificar parámetros esperados
        if !(ptx.contains("param_A") && ptx.contains("param_B") && ptx.contains("param_Out")) {
            errors.push("Missing expected kernel parameters".into());
        }

        // 4. Debe tener una instrucción ret;
        if !ptx.contains("ret;") {
            errors.push("Missing ret instruction".into());
        }

        PtxValidationResult {
            ok: errors.is_empty(),
            errors,
        }
    }
}
