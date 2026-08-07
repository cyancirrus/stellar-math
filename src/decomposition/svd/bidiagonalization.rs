#![allow(unused)]
use crate::decomposition::francis::constants::{EPSILON, MAX_ITERS};
use crate::decomposition::francis::primitives::{
    deflate, lapply_householder, params, rapply_householder,
};
use crate::decomposition::francis::symmetric::francis_iteration_sym;
use crate::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};

fn zero_col(b: &mut [f32], p: &mut [f32], w: &mut [f32], ract: usize, cact: usize, stride: usize) {
    let mut roffset = 0;
    for k in 0..ract {
        w[k] = b[roffset];
        b[roffset] = 0f32;
        roffset += stride;
    }
    let proj = &mut p[..ract];
    let tau = params(&mut w[..ract], proj);
    b[0] = w[0];
    if cact != 0 && tau != 0f32 {
        lapply_householder(
            &mut b[1..],
            proj,
            w,
            tau,
            ract,
            cact.saturating_sub(1),
            stride,
        );
    }
}
fn zero_row(b: &mut [f32], p: &mut [f32], w: &mut [f32], ract: usize, cact: usize, stride: usize) {
    let slice = &mut b[..cact];
    let proj = &mut p[..cact];
    let tau = params(slice, proj);
    if ract != 0 && tau != 0f32 {
        rapply_householder(
            &mut b[stride..],
            proj,
            w,
            tau,
            ract.saturating_sub(1),
            cact,
            stride,
        );
    }
}
/// # ubidiagonal :: upper bidiagonal
///
/// * h: matrix to create the bidiagonal
/// * p: projection vector
/// * w: workspace vector
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
pub fn ubidiagonal(
    b: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
) {
    let mut ract = rows;
    let mut cact = cols;
    let mut o = 0;
    for _ in 0..card.saturating_sub(1) {
        zero_col(&mut b[o..], p, w, ract, cact, stride);
        zero_row(&mut b[o + 1..], p, w, ract - 1, cact - 1, stride);
        ract -= 1;
        cact -= 1;
        o += stride + 1;
    }
    if cact > ract {
        zero_row(&mut b[o..], p, w, ract - 1, cact, stride);
    } else if cact < ract {
        zero_col(&mut b[o..], p, w, ract, cact - 1, stride);
    }
}

/// # lbidiagonal :: lower bidiagonal
///
/// * h: matrix to create the bidiagonal
/// * p: projection vector
/// * w: workspace vector
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
pub fn lbidiagonal(
    b: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
) {
    // rows and active columns
    let mut ract = rows;
    let mut cact = cols;
    let mut o = 0;
    for _ in 0..card.saturating_sub(1) {
        zero_row(&mut b[o..], p, w, ract, cact, stride);
        zero_col(&mut b[o + stride..], p, w, ract - 1, cact, stride);
        o += stride + 1;
        ract -= 1;
        cact -= 1;
    }
    if cact > ract {
        zero_row(&mut b[o..], p, w, ract - 1, cact, stride);
    } else if cact < ract {
        zero_col(&mut b[o..], p, w, ract, cact - 1, stride);
    }
}
#[rustfmt::skip]
pub fn decomp_ugivens(
    h: &mut [f32],
    card: usize,
    stride: usize,
    max_iters:usize,
    threshold: f32,
    absolute: f32,
) {
    let mut curriter=0;
    let mut offset:usize;
    let mut error = f32::INFINITY;
    let mut max_iters:usize=1;
    while threshold < error && curriter < max_iters {
        offset = 0;
        error = 0f32;
        curriter += 1;
        // push zero into col
        let (_, cosine, sine) =
            implicit_givens_rotation(h[0], h[1]);
        apply_gt_right(h, 0, 1, stride, 2, cosine, sine);
        for _ in 0..card.saturating_sub(2) {
            // push zero into row
            let (_, cosine, sine) =
                implicit_givens_rotation(h[offset], h[offset + stride]);
            apply_g_left(&mut h[offset..], 0, 1, stride, 3, cosine, sine);
            // push zero into col
            offset += 1;
            let (_, cosine, sine) =
                implicit_givens_rotation(h[offset], h[offset + 1]);
            apply_gt_right(&mut h[offset ..], 0, 1, stride, 3, cosine, sine);
            error += h[offset].abs();
            offset += stride;
        }
        // push zero into row
        let (_, cosine, sine) =
            implicit_givens_rotation(h[offset], h[offset + stride]);
        apply_g_left(&mut h[offset..], 0, 1, stride, 2, cosine, sine);
        error += h[offset + 1].abs();
    }
}
// #[rustfmt::skip]
// pub fn decomp_lgivens(
//     h: &mut [f32],
//     card: usize,
//     stride: usize,
//     max_iters:usize,
//     tolerance: f32,
//     absolute: f32,
// ) {
//     let mut curriter=0;
//     while curriter < max_iters {
//         curriter += 1;
//         for i in 0..card.saturating_sub(1) {
//             let (_, cosine, sine) =
//                 implicit_givens_rotation(h[i * stride + i], h[(i + 1) * stride + i]);
//             // below diagonal element
//             apply_g_left(h, i, i + 1, stride, card, cosine, sine);

//             let (_, cosine, sine) =
//                 implicit_givens_rotation(h[i * stride + i], h[i * stride + i + 1]);
//             apply_gt_right(h, i, i + 1, stride, card, cosine, sine);
//         }
//     }
// }
