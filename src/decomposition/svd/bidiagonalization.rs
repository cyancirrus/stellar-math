#![allow(unused)]
use crate::decomposition::francis::constants::{EPSILON, MAX_ITERS};
use crate::decomposition::francis::primitives::{
    deflate, lapply_householder, params, rapply_householder,
};
use crate::decomposition::francis::symmetric::francis_iteration_sym;
use crate::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};

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
    let mut offset = 0;
    let mut active_range = rows;
    let mut split_range = cols;
    for o in 0..rows - 1 {
        active_range -= 1;
        split_range -= 1;
        {
            let mut roffset = offset;
            for k in 0..=active_range {
                w[k] = b[roffset + o];
                b[roffset + o] = 0f32;
                roffset += stride;
            }
            let proj = &mut p[..=active_range];
            let tau = params(&mut w[..=active_range], proj);
            b[offset + o] = w[0];
            if tau == 0f32 {
                continue;
            }
            lapply_householder(
                &mut b[offset + o + 1..],
                proj,
                w,
                tau,
                active_range + 1,
                split_range,
                stride,
            );
        }
        {
            let idx = o + 1;
            let slice = &mut b[offset + idx..offset + cols];
            let proj = &mut p[..split_range];
            let tau = params(slice, proj);
            if tau == 0f32 {
                continue;
            }
            rapply_householder(
                &mut b[offset + stride + idx..],
                proj,
                w,
                tau,
                rows - idx,
                split_range,
                stride,
            );
        }
        offset += stride;
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
