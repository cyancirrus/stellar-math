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
    d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    // suffix c: chunk, suffix a: actual
    // let d_add = p - p.min(m) + 1;
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
    d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // suffix c: chunk, suffix a: actual
    // let d_sub = m.saturating_sub(p);
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
    d_add: usize,
    d_sub: usize,
    _m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    // suffix c: chunk, suffix a: actual
    // let d_add = n - n.min(p);
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
    d_add: usize,
    d_sub: usize,
    _m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    // let d_add = p - p.min(n) + 1;
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
    d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    // suffix c: chunk, suffix a: actual
    // let d_add = p - p.min(m) + 1;
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
    d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // diagonal
    // suffix c: chunk, suffix a: actual
    // let d_sub = m.saturating_sub(p);
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
