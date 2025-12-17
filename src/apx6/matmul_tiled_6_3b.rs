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

    // Asegurar que la salida empieza en cero antes de acumular contribuciones
    // de cada bloque en K.
    for v in out.iter_mut() {
        *v = 0.0;
    }

    for i0 in (0..m).step_by(bm) {
        for j0 in (0..n).step_by(bn) {
            for p0 in (0..k).step_by(bk) {
                let i_max = (i0 + bm).min(m);
                let j_max = (j0 + bn).min(n);
                let p_max = (p0 + bk).min(k);

                // Empaquetar B en un panel contiguo [bn x bk] para este tile,
                // en layout column-major respecto a K: para una columna fija j
                // (jj), los distintos pp son contiguos.
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
                        // Índice: jj * bk + pp para que K sea la dimensión contigua.
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

                                // Leer B desde el panel empaquetado: para una
                                // columna fija jj = (j-j0), los 8 valores a lo
                                // largo de K están contiguos.
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

                        // Acumular contribuciones de cada bloque en K sobre out.
                        out[i * n + j] += sum8 + tail;
                    }
                }
            }
        }
    }
}

/// APX 7.0: variante paralela de 6.3b que ejecuta tiles (i0,j0) en
/// paralelo usando PEXExecutor. Cada tarea escribe en una región disjunta
/// de `out`, y el bucle sobre `p0` sigue siendo secuencial dentro de la
/// tarea, preservando la suma exacta por elemento.
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

    // Inicializar la salida a cero igual que en la versión secuencial.
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let threads = crate::cpu_features::cpu_features().threads.max(1) as usize;
    let wsexec = crate::apx7::pex_engine::PEXWorkStealing::new(threads);

    // Compartimos únicamente direcciones base numéricas para a, b y out
    // para evitar capturar referencias no 'static en las closures.
    // Cada tarea escribe en un bloque (i0..i_max, j0..j_max) disjunto,
    // por lo que no hay data races.
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

                    // Panel local de B igual que en la versión secuencial.
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
