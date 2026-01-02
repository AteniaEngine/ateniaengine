#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayer {
    VRAM,
    RAM,
    SSD,
}
