#![allow(unused_imports, dead_code, unused_variables, unused)]
use crate::algebra::ndmethods::basic_mult;
use crate::algebra::ndmethods::create_identity_matrix;
use crate::algebra::ndmethods::matrix_mult;
use crate::decomposition::francis::constants::{MAX_ITERS, TOLERANCE};
use crate::decomposition::lower_upper::LuPivotDecompose;
use crate::decomposition::lq::AutumnDecomp;
use crate::decomposition::schur::real_schur;
use crate::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};
use crate::equality::approximate::approx_vector_tol_eq;
use crate::random::generation::{
    generate_identity_vector, generate_random_matrix, generate_random_vector,
};
#[rustfmt::skip]
use crate::decomposition::francis::primitives::{
    complex_eig_pair,
    deflate,
    eigen,
    double_shift,
    exception_shift,
    lapply_householder,
    params,
    rapply_householder,
};
fn decomp_cpx(h: &mut [f32], w: &mut [f32], mut range: usize, size: usize, mut stride: usize) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(stride + 1);
    let mut e2 = s.saturating_sub(stride + stride + 2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut bl = s.saturating_sub(2);
    let mut curriter = 0;
    let he1 = h[e1];
    let he2 = h[e2];
    let p = &mut [0f32; 3];
    let mut stall = 0;
    while range > 0 && curriter < MAX_ITERS {
        curriter += 1;
        if h[e1].abs() < TOLERANCE {
            stall = 0;
            deflate(
                1,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
        } else if h[e2].abs() < TOLERANCE {
            // if e2 == 0 then we are hitting eigen which should be greater than tolerance
            deflate(
                2,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
            stall = 0;
        } else if range == 2 && complex_eig_pair(h, tl, bl) {
            deflate(
                2,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
            stall = 0;
        } else {
            if range == 2 {
                francis_iteration_cpx_2x2(h, size, stride, tl, bl);
            // } else if stall > 0 && (stall + 4) % 10 == 0 {
            } else if (stall + 8) % 12 == 0 {
                // } else if (stall + 4) % 10 == 0 {
                // } else if stall == 6 {
                exception_shift(h, w, stride, range, tl, bl);
                francis_iteration_cpx(h, p, w, size, range, stride, tl, bl);
            } else {
                double_shift(h, w, stride, range, tl, bl);
                francis_iteration_cpx(h, p, w, size, range, stride, tl, bl);
            }
            stall += 1;
        }
    }
    if range > 1 {
        println!("missed");
    }
}
/// francis_iteration_cpx
///
/// * h: hessenberg linearized matrix
/// * p: projection slice
/// * w: workspace slice
/// * size: static number of rows for rotations
/// * range: number of rows in active window
/// * stride: stride of the data format
fn francis_iteration_cpx(
    h: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    size: usize,
    range: usize,
    stride: usize,
    tl: usize,
    bl: usize,
) {
    let bound = range.min(3);
    let p = &mut p[..bound];
    let tau = params(&mut w[..bound], p);
    if tau != 0f32 {
        rapply_householder(h, p, w, tau, size, bound, stride);
        lapply_householder(h, p, w, tau, bound, range, stride);
        // lapply_householder(h, p, w, tau, bound, size, stride);
    }
    let mut offset = 0;
    for o in 1..range.saturating_sub(1) {
        let bound = bound.min(stride - o);
        let (slice, t) = h.split_at_mut(offset + stride);
        let slice = &mut slice[offset + o..offset + o + bound];
        let proj = &mut p[..bound];
        let tau = params(slice, proj);
        offset += stride;
        if tau == 0f32 {
            continue;
        }
        rapply_householder(&mut t[o..], proj, w, tau, size - o, bound, stride);
        // lapply_householder(&mut h[offset..], proj, w, tau, bound, size, stride);
        lapply_householder(&mut h[offset..], proj, w, tau, bound, range, stride);
    }
}
fn francis_iteration_cpx_2x2(h: &mut [f32], size: usize, stride: usize, tl: usize, bl: usize) {
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[1]);
    apply_g_right(h, 0, 1, stride, size, cosine, -sine);
    apply_gt_left(h, 0, 1, stride, 2, cosine, -sine);
}
