use std::arch::x86_64::*;

pub fn matmul_tiled_6_3b(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let bm = 32;
    let bn = 32;
    let bk = 32;

    // Ensure the output starts at zero before accumulating contributions
    // from each K block.
    for v in out.iter_mut() {
        *v = 0.0;
    }

    for i0 in (0..m).step_by(bm) {
        for j0 in (0..n).step_by(bn) {
            for p0 in (0..k).step_by(bk) {
                let i_max = (i0 + bm).min(m);
                let j_max = (j0 + bn).min(n);
                let p_max = (p0 + bk).min(k);

                // Pack B into a contiguous [bn x bk] panel for this tile, in
                // column-major layout with respect to K: for a fixed column j
                // (jj), different pp values are contiguous.
                let mut b_panel = vec![0f32; bk * bn];
                for pp in 0..bk {
                    let p = p0 + pp;
                    if p >= k {
                        break;
                    }
                    for jj in 0..bn {
                        let j = j0 + jj;
                        if j >= n {
                            break;
                        }
                        // Index: jj * bk + pp so that K is the contiguous dimension.
                        b_panel[jj * bk + pp] = b[p * n + j];
                    }
                }

                for i in i0..i_max {
                    for j in j0..j_max {
                        let mut acc = unsafe { _mm256_setzero_ps() };

                        let mut p = p0;
                        while p + 8 <= p_max {
                            unsafe {
                                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * k + p));

                                // Read B from the packed panel: for a fixed
                                // column jj = (j-j0), the 8 values along K are
                                // contiguous.
                                let panel_col = (j - j0) as usize;
                                let panel_row = (p - p0) as usize;
                                let b_ptr = b_panel
                                    .as_ptr()
                                    .add(panel_col * bk + panel_row);
                                let b_vec = _mm256_loadu_ps(b_ptr);

                                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                            }
                            p += 8;
                        }

                        let mut tail = 0.0f32;
                        for pp in p..p_max {
                            tail += a[i * k + pp] * b[pp * n + j];
                        }

                        let sum8: f32 = unsafe {
                            let r = acc;
                            let hi = _mm256_extractf128_ps(r, 1);
                            let lo = _mm256_castps256_ps128(r);
                            let sum = _mm_add_ps(lo, hi);
                            let shuf = _mm_movehdup_ps(sum);
                            let sum2 = _mm_add_ps(sum, shuf);
                            let shuf2 = _mm_movehl_ps(shuf, sum2);
                            let sum3 = _mm_add_ss(sum2, shuf2);
                            _mm_cvtss_f32(sum3)
                        };

                        // Accumulate contributions from each K block into out.
                        out[i * n + j] += sum8 + tail;
                    }
                }
            }
        }
    }
}

/// APX 7.0: parallel variant of 6.3b that executes tiles (i0,j0) in
/// parallel using PEXExecutor. Each task writes to a disjoint region of
/// `out`, and the loop over `p0` remains sequential within the task,
/// preserving the exact per-element sum.
pub fn matmul_tiled_6_3b_pex(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
){
    let bm = 32;
    let bn = 32;
    let bk = 32;

    // Initialize output to zero, same as the sequential version.
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let threads = crate::cpu_features::cpu_features().threads.max(1) as usize;
    let wsexec = crate::apx7::pex_engine::PEXWorkStealing::new(threads);

    // We only share numeric base addresses for a, b, and out to avoid
    // capturing non-'static references in the closures.
    // Each task writes to a disjoint (i0..i_max, j0..j_max) block,
    // so there are no data races.
    let a_base = a.as_ptr() as usize;
    let b_base = b.as_ptr() as usize;
    let out_base = out.as_mut_ptr() as usize;

    let mut tasks: Vec<Box<dyn FnOnce() + Send>> = Vec::new();

    for i0 in (0..m).step_by(bm) {
        for j0 in (0..n).step_by(bn) {
            let a_addr = a_base;
            let b_addr = b_base;
            let out_addr = out_base;

            let task = Box::new(move || {
                let i_max = (i0 + bm).min(m);
                let j_max = (j0 + bn).min(n);
                let out_p = out_addr as *mut f32;
                let a_ptr = a_addr as *const f32;
                let b_ptr = b_addr as *const f32;

                for p0 in (0..k).step_by(bk) {
                    let p_max = (p0 + bk).min(k);

                    // Local B panel, same as in the sequential version.
                    let mut b_panel = vec![0f32; bk * bn];
                    for pp in 0..bk {
                        let p = p0 + pp;
                        if p >= k {
                            break;
                        }
                        for jj in 0..bn {
                            let j = j0 + jj;
                            if j >= n {
                                break;
                            }
                            unsafe {
                                let src = b_ptr.add(p * n + j);
                                b_panel[jj * bk + pp] = *src;
                            }
                        }
                    }

                    for i in i0..i_max {
                        for j in j0..j_max {
                            let mut acc = unsafe { _mm256_setzero_ps() };

                            let mut p = p0;
                            while p + 8 <= p_max {
                                unsafe {
                                    let a_vec = _mm256_loadu_ps(a_ptr.add(i * k + p));

                                    let panel_col = (j - j0) as usize;
                                    let panel_row = (p - p0) as usize;
                                    let b_panel_ptr = b_panel
                                        .as_ptr()
                                        .add(panel_col * bk + panel_row);
                                    let b_vec = _mm256_loadu_ps(b_panel_ptr);

                                    acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                                }
                                p += 8;
                            }

                            let mut tail = 0.0f32;
                            for pp in p..p_max {
                                unsafe {
                                    let a_val = *a_ptr.add(i * k + pp);
                                    let b_val = *b_ptr.add(pp * n + j);
                                    tail += a_val * b_val;
                                }
                            }

                            let sum8: f32 = unsafe {
                                let r = acc;
                                let hi = _mm256_extractf128_ps(r, 1);
                                let lo = _mm256_castps256_ps128(r);
                                let sum = _mm_add_ps(lo, hi);
                                let shuf = _mm_movehdup_ps(sum);
                                let sum2 = _mm_add_ps(sum, shuf);
                                let shuf2 = _mm_movehl_ps(shuf, sum2);
                                let sum3 = _mm_add_ss(sum2, shuf2);
                                _mm_cvtss_f32(sum3)
                            };

                            unsafe {
                                let idx = i * n + j;
                                *out_p.add(idx) += sum8 + tail;
                            }
                        }
                    }
                }
            });

            tasks.push(task);
        }
    }

    let ws_tasks: Vec<_> = tasks
        .into_iter()
        .map(|t| crate::apx7::pex_engine::PEXTask::Tile(t))
        .collect();

    wsexec.execute_parallel_ws(ws_tasks);
}
