#![allow(unused)]
use crate::decomposition::francis::constants::{EPSILON, MAX_ITERS};
use crate::decomposition::francis::primitives::{
    deflate, lapply_householder, params, rapply_householder,
};
use crate::decomposition::francis::symmetric::francis_iteration_sym;
use crate::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};
#[rustfmt::skip]
pub fn decomp_ugivens(
    h: &mut [f32],
    card: usize,
    stride: usize,
    max_iters:usize,
    threshold: f32,
    absolute: f32,
) {
    let interior = card.saturating_sub(2);
    let mut supdiag_norm = f32::INFINITY;
    for _ in 0..max_iters {
        if supdiag_norm < threshold { break; }
        let mut offset = 0;
        // push zero into col
        let (_, cos, sin) = implicit_givens_rotation(h[0], h[1]);
        apply_gt_right(h, 0, 1, stride, 2, cos, sin);
        for _ in 0..interior {
            // push zero into row
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
            apply_g_left(&mut h[offset..], 0, 1, stride, 3, cos, sin);
            // push zero into col
            offset += 1;
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
            apply_gt_right(&mut h[offset ..], 0, 1, stride, 3, cos, sin);
            supdiag_norm += h[offset].abs();
            offset += stride;
        }
        // push zero into row
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
        apply_g_left(&mut h[offset..], 0, 1, stride, 2, cos, sin);
        supdiag_norm += h[offset + 1].abs();
    }
}
#[rustfmt::skip]
pub fn decomp_lgivens(
    h: &mut [f32],
    card: usize,
    stride: usize,
    max_iters:usize,
    threshold: f32,
    absolute: f32,
) {
    let interior = card.saturating_sub(2);
    let mut subdiag_norm = f32::INFINITY;
    for _ in 0..max_iters {
        if subdiag_norm < threshold { break; }
        subdiag_norm = 0f32;
        let mut offset = 0;
        // push zero into row
        let (_, cos, sin) = implicit_givens_rotation(h[0], h[stride]);
        apply_g_left(h, 0, 1, stride, 2, cos, sin);
        for _ in 0..interior {
            // push zero into col
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
            apply_gt_right(&mut h[offset ..], 0, 1, stride, 3, cos, sin);
            // push zero into row
            offset += stride;
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
            apply_g_left(&mut h[offset..], 0, 1, stride, 3, cos, sin);
            subdiag_norm += h[offset].abs();
            offset += 1;
        }
        // // push zero into col
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
        apply_gt_right(&mut h[offset ..], 0, 1, stride, 2, cos, sin);
        subdiag_norm += h[offset + stride].abs();
    }
}
#[rustfmt::skip]
pub fn full_decomp_lgivens(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
    max_iters:usize,
    threshold: f32,
    absolute: f32,
) {
    let interior = card.saturating_sub(2);
    let mut subdiag_norm = f32::INFINITY;
    // for _ in 0..max_iters {
    for _ in 0..1 {
        if subdiag_norm < threshold { break; }
        subdiag_norm = 0f32;
        let mut offset = 0;
        let mut uoffset = 0;
        let mut voffset = 0;
        // push zero into row
        let (_, cos, sin) = implicit_givens_rotation(h[0], h[stride]);
        apply_g_left(h, 0, 1, stride, 2, cos, sin);
        apply_gt_right(u, 0, 1, rows, rows, cos, sin);
        for _ in 0..interior {
            // push zero into col
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
            apply_gt_right(&mut h[offset ..], 0, 1, stride, 3, cos, sin);
            apply_gt_right(&mut v[voffset ..], 0, 1, cols, cols, cos, sin);
            // push zero into row
            offset += stride;
            voffset += 1;
            uoffset += 1;

            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
            apply_g_left(&mut h[offset..], 0, 1, stride, 3, cos, sin);
            apply_gt_right(&mut u[uoffset..], 0, 1, rows, rows, cos, sin);
            subdiag_norm += h[offset].abs();
            offset += 1;
        }
        // // push zero into col
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
        apply_gt_right(&mut h[offset ..], 0, 1, stride, 2, cos, sin);
        apply_gt_right(&mut v[voffset ..], 0, 1, cols, cols, cos, sin);
        subdiag_norm += h[offset + stride].abs();
    }
}
#[rustfmt::skip]
pub fn full_decomp_ugivens(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
    max_iters:usize,
    threshold: f32,
    absolute: f32,
) {
    let interior = card.saturating_sub(2);
    let mut supdiag_norm = f32::INFINITY;
    for _ in 0..max_iters {
        if supdiag_norm < threshold { break; }
        let mut offset = 0;
        let mut uoffset = 0;
        let mut voffset = 0;
        // push zero into col
        let (_, cos, sin) = implicit_givens_rotation(h[0], h[1]);
        apply_gt_right(h, 0, 1, stride, 2, cos, sin);
        apply_gt_right(v, 0, 1, cols, cols, cos, sin);
        for _ in 0..interior {
            // push zero into row
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
            apply_g_left(&mut h[offset..], 0, 1, stride, 3, cos, sin);
            apply_gt_right(&mut u[uoffset..], 0, 1, rows, rows, cos, sin);
            // push zero into col
            offset += 1;
            voffset += 1;
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
            apply_gt_right(&mut h[offset ..], 0, 1, stride, 3, cos, sin);
            apply_gt_right(&mut v[voffset ..], 0, 1, cols, cols, cos, sin);
            supdiag_norm += h[offset].abs();
            uoffset += 1;
            offset += stride;
        }
        // push zero into row
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
        apply_g_left(&mut h[offset..], 0, 1, stride, 2, cos, sin);
        apply_gt_right(&mut u[uoffset..], 0, 1, rows, rows, cos, sin);
        supdiag_norm += h[offset + 1].abs();
    }
}
