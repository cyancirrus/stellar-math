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
            //NOTE: THINK THIS SHOULD BE RACT
            ract.saturating_sub(1),
            cact,
            stride,
        );
    }
}
fn full_zero_col(
    b: &mut [f32],
    u: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    ract: usize,
    cact: usize,
    stride: usize,
) {
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
        rapply_householder(
            u,
            proj,
            w,
            tau,
            rows,
            ract,
            rows,
        );
    }
}
fn full_zero_row(
    b: &mut [f32],
    v: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    cols: usize,
    ract: usize,
    cact: usize,
    stride: usize,
) {
    let slice = &mut b[..cact];
    let proj = &mut p[..cact];
    let tau = params(slice, proj);
    if ract != 0 && tau != 0f32 {
        rapply_householder(
            &mut b[stride..],
            proj,
            w,
            tau,
            ract,
            cact,
            stride,
        );
        w.fill(0f32);
        rapply_householder(
            // &mut v[stride..],
            &mut v[..],
            proj,
            w,
            tau,
            cols,
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
/// * b: matrix to create the bidiagonal
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
/// # full ubidiagonal :: upper bidiagonal
///
/// * b: matrix to create the bidiagonal
/// * u: eigenvectors of AA' ie rowspace
/// * v: eigenvectors of A'A ie colspace
/// * p: projection vector
/// * w: workspace vector
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
#[rustfmt::skip]
pub fn full_ubidiagonal(
    b: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
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
    // for k in 0..card.saturating_sub(1) {
    for k in 0..1 {
    // for k in 0..2 {
        println!("hello");
        full_zero_col(&mut b[o + k..], &mut u[k..], p, w, rows, ract, cact, stride);
        // full_zero_row(&mut b[o + k + 1..], &mut v[k + 1 ..], p, w, cols, ract - 1, cact - 1, stride);
        ract -= 1;
        cact -= 1;
        o += stride;
    }
    // if cact < ract {
    //     full_zero_col(&mut b[o..], u, p, w, rows, ract, cact - 1, stride);
    // } else if cact > ract {
    //     full_zero_row(&mut b[o..], v, p, w, cols, ract - 1, cact, stride);
    // }
}
