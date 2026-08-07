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
    println!("0COL    XXX ract, cact (ract {ract:}, {cact:})");
    let proj = &mut p[..ract];
    let tau = params(&mut w[..ract], proj);
    b[0] = w[0];
    if tau != 0f32 && cact != 0 {
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
    if tau != 0f32 && ract != 0 {
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
    stride: usize,
) {
    let mut ract = rows;
    let mut cact = cols;
    let mut pivot = rows.min(cols).saturating_sub(1);
    let mut o= 0;
    for _ in 0..pivot {
        zero_col(&mut b[o..], p, w, ract, cact, stride);
        zero_row(&mut b[o+1..], p, w, ract - 1, cact - 1, stride);
        ract -= 1;
        cact -= 1;
        o += stride + 1;
    }
    if cact > ract {
        zero_row(&mut b[o..], p, w, ract - 1 , cact, stride);
    } else if cact < ract {
        zero_col(&mut b[o..], p, w, ract , cact - 1, stride);
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
    stride: usize,
) {
    // rows and active columns
    let mut ract = rows;
    let mut cact = cols;
    let mut card = rows.min(cols);
    let mut o = 0;
    for _ in 0..card.saturating_sub(1) {
        zero_row(&mut b[o ..], p, w, ract, cact, stride);
        zero_col(&mut b[o + stride..], p, w, ract - 1, cact, stride);
        o += stride + 1;
        ract -= 1;
        cact -= 1;
    }
    if cact > ract {
        zero_row(&mut b[o..], p, w, ract - 1 , cact, stride);
    } else if cact < ract {
        zero_col(&mut b[o..], p, w, ract , cact - 1, stride);
    }
}
#[rustfmt::skip]
pub fn decomp_givens(
    h: &mut [f32],
    mut range: usize,
    size: usize,
    stride: usize,
    max_iters:usize,
    tolerance: f32,
    absolute: f32,
) {
    let mut curriter=0;
    // left work

    while curriter < max_iters {
        curriter += 1;
        for i in 0..range - 1 {
            let (_, cosine, sine) =
                implicit_givens_rotation(h[i * stride + i], h[(i + 1) * stride + i]);
            // below diagonal element
            apply_g_left(h, i, i + 1, stride, range, cosine, sine);

            let (_, cosine, sine) =
                implicit_givens_rotation(h[i * stride + i], h[i * stride + i + 1]);
            apply_gt_right(h, i, i + 1, stride, range, cosine, sine);
        }
    }
}
