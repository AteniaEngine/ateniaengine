// APX 9.3 — PTX Validator v0
// Purely textual PTX syntactic/structural validator.
// Does not execute nor compile PTX; does not require CUDA nor external toolchains.

pub struct PtxValidationResult {
    pub ok: bool,
    pub errors: Vec<String>,
}

pub struct PtxValidator;

impl PtxValidator {
    pub fn validate(ptx: &str) -> PtxValidationResult {
        let mut errors = Vec::new();

        // 1. Must contain .version
        if !ptx.contains(".version") {
            errors.push("Missing .version directive".into());
        }

        // 2. Must contain .entry
        if !ptx.contains(".entry") {
            errors.push("Missing .entry function".into());
        }

        // 3. Verify expected parameters
        if !(ptx.contains("param_A") && ptx.contains("param_B") && ptx.contains("param_Out")) {
            errors.push("Missing expected kernel parameters".into());
        }

        // 4. Must contain a ret instruction;
        if !ptx.contains("ret;") {
            errors.push("Missing ret instruction".into());
        }

        PtxValidationResult {
            ok: errors.is_empty(),
            errors,
        }
    }
}
