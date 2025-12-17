#[cfg(debug_assertions)]
pub fn log(msg: &str) {
    println!("[AGR] {}", msg);
}

#[cfg(not(debug_assertions))]
pub fn log(_msg: &str) {}
