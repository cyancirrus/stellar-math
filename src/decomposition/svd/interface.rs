use crate::decomposition::svd::bidiagonalization::{ubidiagonal, lbidiagonal};
use crate::decomposition::svd::bidiagonalization::{full_ubidiagonal, full_lbidiagonal};
use crate::decomposition::svd::bulge_chasing::{full_decomp_ugivens, full_decomp_lgivens};
use crate::decomposition::svd::bulge_chasing::{decomp_ugivens, decomp_lgivens};

pub fn full_svd_decomposition(
    b: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
    max_iters: usize,
    threshold: f32,
    absolute: f32,
) {
    if cols > rows {
        full_lbidiagonal(b, u, v, p, w, rows, cols, card, stride);
        full_decomp_lgivens(b, u, v, rows, cols, card, stride, max_iters, threshold, absolute);

    } else {
        full_ubidiagonal(b, u, v, p, w, rows, cols, card, stride);
        full_decomp_ugivens(b, u, v, rows, cols, card, stride, max_iters, threshold, absolute);
    }
}

pub fn svd_decomposition(
    b: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
    max_iters: usize,
    threshold: f32,
    absolute: f32,
) {
    if cols > rows {
        lbidiagonal(b, p, w, rows, cols, card, stride);
        decomp_lgivens(b, card, stride, max_iters, threshold, absolute);
    } else {
        ubidiagonal(b, p, w, rows, cols, card, stride);
        decomp_ugivens(b, card, stride, max_iters, threshold, absolute);
    }
}
