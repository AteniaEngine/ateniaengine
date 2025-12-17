// APX 9.23 — Dual-Issue SM (SIMT, totalmente simulado)
// Implementa un scheduler dual-issue simbólico que opera sobre VGPUWarp y VGPUScoreboard.
// No toca kernels reales ni backward, sólo estructuras internas de simulación.

use crate::apx9::vgpu_warp::VGPUWarp;
use crate::apx9::vgpu_scoreboard::VGPUScoreboard;
use crate::apx9::vgpu_instr::VGPUInstr;

#[derive(Debug)]
pub struct VGPUDualIssue;

impl VGPUDualIssue {
    pub fn issue(
        warp: &mut VGPUWarp,
        instr1: Option<VGPUInstr>,
        instr2: Option<VGPUInstr>,
    ) -> (bool, bool) {
        let mut ok1 = false;
        let mut ok2 = false;

        // Issue slot #1
        if let Some(i1) = instr1.clone() {
            if !Self::has_hazard(&i1, &warp.scoreboard) {
                // Reservamos en el scoreboard del warp.
                warp.scoreboard.reserve(&i1);
                warp.pipeline.push(i1);
                ok1 = true;
            }
        }

        // Issue slot #2 (evitar conflictos RAW/WAW con i1)
        if let Some(i2) = instr2.clone() {
            if !Self::has_hazard(&i2, &warp.scoreboard)
                && !Self::conflicts(instr1.as_ref(), &i2)
            {
                warp.scoreboard.reserve(&i2);
                warp.pipeline.push(i2);
                ok2 = true;
            }
        }

        (ok1, ok2)
    }

    fn has_hazard(instr: &VGPUInstr, sb: &VGPUScoreboard) -> bool {
        instr.read_regs().iter().any(|r| !sb.can_read(*r))
            || instr.write_regs().iter().any(|r| !sb.can_write(*r))
    }

    fn conflicts(i1: Option<&VGPUInstr>, i2: &VGPUInstr) -> bool {
        if let Some(a) = i1 {
            for w in a.write_regs() {
                if i2.read_regs().contains(&w) || i2.write_regs().contains(&w) {
                    return true;
                }
            }
        }
        false
    }
}
