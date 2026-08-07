#![allow(unused)]
use crate::decomposition::francis::constants::{EPSILON, MAX_ITERS};
use crate::decomposition::francis::primitives::{
    deflate, lapply_householder, params, rapply_householder,
};
use crate::decomposition::francis::symmetric::francis_iteration_sym;
use crate::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};


fn zero_col(
    b: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rrange:usize,
    crange:usize,
    stride:usize
) {
    let mut roffset = 0;
    for k in 0..=rrange {
        w[k] = b[roffset];
        b[roffset] = 0f32;
        roffset += stride;
    }
    println!("active_range {rrange:}");
    let proj = &mut p[..=rrange];
    let tau = params(&mut w[..=rrange], proj);
    b[0] = w[0];
    if tau != 0f32 {
        lapply_householder(
            &mut b[1..],
            proj,
            w,
            tau,
            rrange + 1,
            crange,
            stride,
        );
    }
}
fn zero_row(
    b: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rrange:usize,
    crange:usize,
    stride:usize,
) {
    let slice = &mut b[..crange];
    let proj = &mut p[..rrange];
    println!("zeroing_row.. crange {crange:}");
    println!("slice {slice:?}, proj {proj:?}");
    let tau = params(slice, proj);
    if tau != 0f32 {
        rapply_householder(
            &mut b[stride..],
            proj,
            w,
            tau,
            rrange,
            crange,
            stride,
        );
    }
}

/// bidiagonal
/// * h: matrix to create the bidiagonal
/// * p: projection vector
/// * w: workspace vector
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
pub fn bidiagonal(
    b: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    stride: usize,
) {
    // stores tau
    let mut rrange = rows.saturating_sub(1);
    // let mut crange = cols.saturating_sub(1);
    let mut card = rows.min(cols);
    let mut submatrix = b;
    // for o in 0..card.saturating_sub(1) {
    for o in 0..1 {
        // zero_row(submatrix, p, w, rrange, crange, stride,);
        // zero_col(&mut submatrix[1..], p, w, rrange, crange, stride);
        zero_col(submatrix, p, w, rrange, crange, stride);
        zero_row(&mut submatrix[1..], p, w, rrange, crange, stride,);
        submatrix = &mut submatrix[stride + 1..];
        rrange -= 1;
        crange -= 1;
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
