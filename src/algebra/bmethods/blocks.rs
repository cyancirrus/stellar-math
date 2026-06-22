#![allow(unused)]
use crate::algebra::bmethods::primitives::{diff_min, pack};
use crate::arch::SIMD_WIDTH;
use crate::kernel::matkerns::{kernel_mult, kernel_tmult};
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
#[rustfmt::skip]
use crate::algebra::bmethods::contractions::{
    tensor_contraction,
    tensor_tcontraction,
    tensor_lt_contraction,
    tensor_rlt_contraction,
    tensor_rut_contraction,
    tensor_tlt_contraction,
    tensor_tut_contraction,
    tensor_ut_contraction,
};

const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
// FASTEST
const MC: usize = 40;
const PC: usize = 160;
const NC: usize = 120;

thread_local! {
    static PACK: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> = RefCell::new((vec![0f32; MC * PC], vec![0f32; PC * NC], vec![0f32; MC * NC]));
}
pub fn tensor_block(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    _m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // suffix c: chunk, suffix a: actual
    t_d.par_chunks_mut(MC * n)
        .zip(x_d.par_chunks(MC * p))
        .for_each(|(t, x)| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let dy = PC * s_y;
                let (xend, mut yend, tend);
                let rows = x.len() / s_x;
                let ma = rows;
                (xend, tend) = (ma * s_x, ma * s_t);
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        yend = pa * s_y;
                        pack(&x[pc..xend], x_pack, ma, pa, PC, s_x);
                        pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                        tensor_contraction(&x_pack, &y_pack, t_accum, ma, pa, na, PC, NC, NC);
                        yoffset += dy;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_tblock(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // suffix c: chunk, suffix a: actual
    t_d.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(mc_idx, t)| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let dy = PC * s_y;
                let d_xt = PC * s_x;
                let ma = diff_min(m, mc_idx * MC, MC);
                let tend = ma * s_t;
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    let mut xoffset = mc_idx * MC;
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        pack(&x_d[xoffset..], x_pack, pa, ma, MC, s_x);
                        pack(&y_d[yoffset + nc..], y_pack, pa, na, NC, s_y);
                        tensor_tcontraction(&x_pack, &y_pack, t_accum, ma, pa, na, MC, NC, NC);
                        yoffset += dy;
                        xoffset += d_xt;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_lt_block(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    // suffix c: chunk, suffix a: actual
    let d_add = p - p.min(m) + 1;
    t_d.par_chunks_mut(MC * n)
        .zip(x_d.par_chunks(MC * p))
        .enumerate()
        .for_each(|(mc_idx, (t, x))| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let d_add = d_add + mc_idx * MC;
                let dy = PC * s_y;
                let ma = x.len() / s_x;
                let (xend, tend) = (ma * s_x, ma * s_t);
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        let yend = pa * s_y;
                        pack(&x[pc..xend], x_pack, ma, pa, PC, s_x);
                        pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                        tensor_lt_contraction(
                            &x_pack, &y_pack, t_accum, d_add, pc, ma, pa, na, PC, NC, NC,
                        );
                        yoffset += dy;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_ut_block(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // suffix c: chunk, suffix a: actual
    let d_sub = m.saturating_sub(p);
    t_d.par_chunks_mut(MC * n)
        .zip(x_d.par_chunks(MC * p))
        .enumerate()
        .for_each(|(mc_idx, (t, x))| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let d_add = mc_idx * MC;
                let dy = PC * s_y;
                let ma = x.len() / s_x;
                let (xend, tend) = (ma * s_x, ma * s_t);
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        let yend = pa * s_y;
                        pack(&x[pc..xend], x_pack, ma, pa, PC, s_x);
                        pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                        tensor_ut_contraction(
                            &x_pack,
                            &y_pack,
                            t_accum,
                            d_add,
                            d_sub + pc,
                            ma,
                            pa,
                            na,
                            PC,
                            NC,
                            NC,
                        );
                        yoffset += dy;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_rlt_block(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    _m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    // suffix c: chunk, suffix a: actual
    let d_add = n - n.min(p);
    t_d.par_chunks_mut(MC * n)
        .zip(x_d.par_chunks(MC * p))
        .for_each(|(t, x)| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let dy = PC * s_y;
                let ma = x.len() / s_x;
                let (xend, tend) = (ma * s_x, ma * s_t);
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        let yend = pa * s_y;
                        pack(&x[pc..xend], x_pack, ma, pa, PC, s_x);
                        pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                        tensor_rlt_contraction(
                            &x_pack,
                            &y_pack,
                            t_accum,
                            d_add + pc,
                            nc,
                            ma,
                            pa,
                            na,
                            PC,
                            NC,
                            NC,
                        );
                        yoffset += dy;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_rut_block(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    _m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    let d_add = p - p.min(n) + 1;
    t_d.par_chunks_mut(MC * n)
        .zip(x_d.par_chunks(MC * p))
        .for_each(|(t, x)| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let dy = PC * s_y;
                let ma = x.len() / s_x;
                let (xend, tend) = (ma * s_x, ma * s_t);
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        let yend = pa * s_y;
                        pack(&x[pc..xend], x_pack, ma, pa, PC, s_x);
                        pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                        tensor_rut_contraction(
                            &x_pack,
                            &y_pack,
                            t_accum,
                            d_add + nc,
                            pc,
                            ma,
                            pa,
                            na,
                            PC,
                            NC,
                            NC,
                        );
                        yoffset += dy;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_tlt_block(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    // suffix c: chunk, suffix a: actual
    let d_add = p - p.min(m) + 1;
    t_d.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(mc_idx, t)| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let d_add = d_add + mc_idx * MC;
                let dy = PC * s_y;
                let d_xt = PC * s_x;
                let ma = diff_min(m, mc_idx * MC, MC);
                // let (xend, tend) = (ma * s_x, ma * s_t);
                let tend = ma * s_t;
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    // base column offset
                    let mut xoffset = mc_idx * MC;
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        // let yend = pa * s_y;
                        pack(&x_d[xoffset..], x_pack, pa, ma, MC, s_x);
                        // pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                        pack(&y_d[yoffset + nc..], y_pack, pa, na, NC, s_y);
                        tensor_tlt_contraction(
                            &x_pack, &y_pack, t_accum, d_add, pc, ma, pa, na, MC, NC, NC,
                        );
                        yoffset += dy;
                        xoffset += d_xt;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_tut_block(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    // suffix c: chunk, suffix a: actual
    let d_sub = m.saturating_sub(p);
    t_d.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(mc_idx, t)| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let d_add = mc_idx * MC;
                let dy = PC * s_y;
                let d_xt = PC * s_x;
                let ma = diff_min(m, mc_idx * MC, MC);
                let tend = ma * s_t;
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    // base column offset
                    let mut xoffset = mc_idx * MC;
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        pack(&x_d[xoffset..], x_pack, pa, ma, MC, s_x);
                        pack(&y_d[yoffset + nc..], y_pack, pa, na, NC, s_y);
                        tensor_tut_contraction(
                            &x_pack,
                            &y_pack,
                            t_accum,
                            d_add,
                            d_sub + pc,
                            ma,
                            pa,
                            na,
                            MC,
                            NC,
                            NC,
                        );
                        yoffset += dy;
                        xoffset += d_xt;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}

#[cfg(test)]
mod test_kernel_block {
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::arch::SIMD_WIDTH;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;

    #[test]
    fn test_outkern_equivalence() {
        let ikj = [
            (1, 1, 1),
            (8, 1, 1),
            (1, 8, 1),
            (1, 1, 8),
            (6, 4, 8),
            (6, 8, 4),
            (4, 6, 8),
            (4, 8, 6),
            (8, 4, 6),
            (8, 6, 4),
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
        ];
        for (i, k, j) in ikj {
            test_outkern_equivalence_mkn(i, k, j);
        }
    }
    fn test_outkern_equivalence_mkn(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut result = vec![0f32; m * n];
        let expected = basic_mult(&x, &y);
        tensor_contraction(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        // let inspect = NdArray {
        //     dims: vec![m, n],
        //     data: result.clone(),
        // };
        // println!("expected {expected:?}");
        // println!("actual {inspect:?}");
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    #[test]
    fn test_gemm_equivalence() {
        let ikj = [
            // (256, 256, 256),
            (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
            (1, 1, 1),
            (8, 1, 1),
            (1, 8, 1),
            (1, 1, 8),
            (6, 4, 8),
            (6, 8, 4),
            (4, 6, 8),
            (4, 8, 6),
            (8, 4, 6),
            (8, 6, 4),
            (8, 8, 8),
            (16, 16, 16),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
            (256, 1024, 512),
            (512, 512, 512),
            (1024, 64, 1024),
        ];
        for (i, k, j) in ikj {
            println!("(i: {i:?}, k: {k:?}, j: {j:})");
            matmul_equivalence(i, k, j);
            tmatmul_equivalence(i, k, j);
        }
    }
    fn matmul_equivalence(m: usize, k: usize, n: usize) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);
        let mut result = vec![0f32; m * n];
        let expected = basic_mult(&x, &y);
        tensor_block(&x.data, &y.data, &mut result, m, k, n, k, n, n);
        let inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        println!("expected {expected:?}");
        println!("actual {inspect:?}");
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    fn tmatmul_equivalence(m: usize, p: usize, n: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        x.transpose_inplace();
        // println!("x_base {x_base:?}");
        // println!("y {y:?}");
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        // m, n, n b/c X is s tored in it's transposed state
        tensor_tblock(&x.data, &y.data, &mut result, m, p, n, m, n, n);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        // println!("expected {expected:?}");
        // println!("actual {_inspect:?}");
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
}
#[cfg(test)]
#[cfg(feature = "avx2")]
mod test_lower_and_upper_triangular_dispatch {
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;
    #[test]
    fn test_gemm_equivalence() {
        let ikj = [
            (9, 16, 9),
            (32, 32, 32),
            (1, 1, 1),
            (16, 16, 16),
            (8, 9, 8),
            (3, 9, 1),
            (6, 4, 8),
            (9, 16, 8),
            (8, 8, 9),
            (2, 9, 1),
            (2, 2, 1),
            (2, 9, 1),
            (2, 10, 1),
            (1, 9, 1),
            (4, 8, 1),
            (1, 2, 1),
            (1, 1, 1),
            (8, 1, 1),
            (1, 8, 1),
            (1, 1, 8),
            (6, 4, 8),
            (6, 8, 4),
            (8, 4, 6),
            (4, 8, 6),
            (4, 6, 8),
            (8, 6, 4),
            (8, 8, 8),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
            (MC + 1, PC, NC + 1),
            (MC + 1, PC, NC - 1),
            (MC + 1, PC, NC),
            (MC - 1, PC, NC),
            (MC, PC + 1, NC),
            (MC, PC - 1, NC),
            (MC, PC, NC),
            (256, 256, 256),
            (256, 1024, 512),
            (512, 512, 512),
            (1024, 64, 1024),
        ];
        for (i, k, j) in ikj {
            println!("(i: {i:?}, k: {k:?}, j: {j:})");
            lower_equivalence_mkn(i, k, j);
            upper_equivalence_mkn(i, k, j);
            rlower_equivalence_mkn(i, k, j);
            rupper_equivalence_mkn(i, k, j);
            ltl_equivalence_mkn(i, k, j);
            ltu_equivalence_mkn(i, k, j);
        }
    }
    /// Case 1 / Case 2
    /// -------+---------
    /// * * *  / * * * *
    /// * * *  / 0 * * *
    /// 0 * *  /
    /// 0 0 *  /
    fn filter_upper_trapezoid(a: &mut NdArray) {
        // i - (m - n);
        let (m, n) = (a.dims[0], a.dims[1]);
        let data = &mut a.data;
        let mut d_sub = if m > n { m - n } else { 0 };
        let mut d_plus = 0;
        for i in 0..m {
            for j in 0..n {
                if j + d_sub >= d_plus {
                    break;
                }
                data[i * n + j] = 0f32;
            }
            d_plus += 1;
        }
    }
    /// * * * * * * * 0 0 0
    /// * * * * * * * * 0 0
    /// * * * * * * * * * 0
    /// * * * * * * * * * *
    fn filter_lower_trapezoid(a: &mut NdArray) {
        let (rows, cols) = (a.dims[0], a.dims[1]);
        let d = &mut a.data;
        let t = cols.min(rows);
        let s = rows.saturating_sub(cols);
        // don't remove from last row
        for i in 1..t {
            for j in 0..i {
                d[(rows - i - s) * cols - j - 1] = 0f32;
            }
        }
    }
    fn lower_equivalence_mkn(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut x_base = x.clone();
        filter_lower_trapezoid(&mut x_base);
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        tensor_lt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn upper_equivalence_mkn(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut x_base = x.clone();
        filter_upper_trapezoid(&mut x_base);
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        tensor_ut_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn rlower_equivalence_mkn(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut y_base = y.clone();
        filter_lower_trapezoid(&mut y_base);
        // println!("x_base {x:?}");
        // println!("y_base {y_base:?}");
        let expected = basic_mult(&x, &y_base);
        let mut result = vec![0f32; m * n];
        tensor_rlt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn rupper_equivalence_mkn(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut y_base = y.clone();
        filter_upper_trapezoid(&mut y_base);
        // println!("x_base {x:?}");
        // println!("y_base {y_base:?}");
        let expected = basic_mult(&x, &y_base);
        let mut result = vec![0f32; m * n];
        tensor_rut_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        // println!("expected {expected:?}");
        // println!("actual {_inspect:?}");
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn ltl_equivalence_mkn(m: usize, p: usize, n: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        filter_lower_trapezoid(&mut x_base);
        x.transpose_inplace(); // stored in transpose
        // println!("x_base {x_base:?}");
        // println!("y {y:?}");
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        // tensor_tlt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        // m, n, n b/c X is s tored in it's transposed state
        tensor_tlt_block(&x.data, &y.data, &mut result, m, p, n, m, n, n);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        // println!("expected {expected:?}");
        // println!("actual {_inspect:?}");
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn ltu_equivalence_mkn(m: usize, p: usize, n: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        // filter_lower_trapezoid(&mut x_base);
        filter_upper_trapezoid(&mut x_base);
        x.transpose_inplace(); // stored in transpose
        // println!("x_base {x_base:?}");
        // println!("y {y:?}");
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        // m, n, n b/c X is s tored in it's transposed state
        tensor_tut_block(&x.data, &y.data, &mut result, m, p, n, m, n, n);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        // println!("expected {expected:?}");
        // println!("actual {_inspect:?}");
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
}
