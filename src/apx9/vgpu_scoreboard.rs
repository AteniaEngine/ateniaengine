// APX 9.21 / 9.23 — SIMT Scoreboard (Hazard & Dependency Manager)
// Dependency model fully simulated on CPU; does not touch real GPU nor backward.

use crate::apx9::vgpu_instr::VGPUInstr;

#[derive(Debug, Clone)]
pub struct VGPUScoreboard {
    /// reg_busy[i] = true -> the register has a pending write.
    pub reg_busy: Vec<bool>,
}

impl VGPUScoreboard {
    pub fn new(num_regs: usize) -> Self {
        Self { reg_busy: vec![false; num_regs] }
    }

    pub fn mark_write(&mut self, reg: usize) {
        if reg < self.reg_busy.len() {
            self.reg_busy[reg] = true;
        }
    }

    pub fn mark_write_complete(&mut self, reg: usize) {
        if reg < self.reg_busy.len() {
            self.reg_busy[reg] = false;
        }
    }

    pub fn can_read(&self, reg: usize) -> bool {
        reg >= self.reg_busy.len() || !self.reg_busy[reg]
    }

    pub fn can_write(&self, reg: usize) -> bool {
        reg >= self.reg_busy.len() || !self.reg_busy[reg]
    }

    /// Mark all registers as free.
    pub fn clear_all(&mut self) {
        for b in &mut self.reg_busy {
            *b = false;
        }
    }

    /// Reserve destination registers of an instruction as "pending write".
    pub fn reserve(&mut self, instr: &VGPUInstr) {
        for dst in instr.write_regs() {
            self.mark_write(dst);
        }
    }
}
