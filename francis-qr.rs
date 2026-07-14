#![allow(unused_imports, dead_code, unused_variables)]
use stellar::algebra::ndmethods::basic_mult;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::equality::approximate::approx_vector_tol_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
// use stellar::decomposition::lq::params;
const TOLERANCE: f32 = 1e-2;

struct FrancisQr {
    kernel: NdArray,
    transform: NdArray,
}
fn eigen(m00: f32, m01: f32, m10: f32, m11: f32) -> f32 {
    let d = (m00 - m11) / 2f32;
    m11 + d - d.signum() * (d * d + m10 * m10).sqrt()
}
// h:= hessenberg matrix
fn decomp(h: &mut [f32], mut range: usize, stride: usize) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(2);
    let mut e2 = s.saturating_sub(stride + 3);
    let mut eig: f32;
    while range != 0 {
        if h[e1].abs() < TOLERANCE {
            range -= 1;
            e1 = e1.saturating_sub(stride + 1);
            e2 = e2.saturating_sub(stride + 1);
        } else if h[e2].abs() < TOLERANCE {
            // if e2 == 0 then we are hitting eigen which should be greater than tolerance
            range -= 2;
            e1 = e1.saturating_sub(2 * stride + 2);
            e2 = e2.saturating_sub(2 * stride + 2);
        } else {
            francis_iteration(h, range, stride);
        }
    }
}
fn francis_iteration(h: &mut [f32], range: usize, stride: usize) {
    let card = stride * range;
    let tl = card.saturating_sub(stride + 2);
    let bl = card.saturating_sub(2);
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[stride]);
    apply_g_left(h, 0, 1, stride, range, cosine, sine);
    apply_gt_right(h, 0, 1, stride, range, cosine, sine);
    for k in 1..range {
        let r = k * stride;
        let (_, cosine, sine) = implicit_givens_rotation(h[r + k], h[r + k + stride]);
        apply_g_left(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
        apply_gt_right(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
    }
}
fn full_decomp(h: &mut [f32], t: &mut [f32], mut range: usize, stride: usize) {
    let card = range;
    let s = range * stride;
    let mut e1 = s.saturating_sub(2);
    let mut e2 = s.saturating_sub(stride + 3);
    let mut eig: f32;
    while range != 0 {
        if h[e1].abs() < TOLERANCE {
            range -= 1;
            e1 = e1.saturating_sub(stride + 1);
            e2 = e2.saturating_sub(stride + 1);
        } else if h[e2].abs() < TOLERANCE {
            // if e2 == 0 then we are hitting eigen which should be greater than tolerance
            range -= 2;
            e1 = e1.saturating_sub(2 * stride + 2);
            e2 = e2.saturating_sub(2 * stride + 2);
        } else {
            full_francis_iteration(h, t, card, range, stride);
        }
    }
}
fn full_francis_iteration(h: &mut [f32], t: &mut [f32], card:usize, range: usize, stride: usize) {
    let card = stride * range;
    let tl = card.saturating_sub(stride + 2);
    let bl = card.saturating_sub(2);
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[stride]);
    apply_g_left(h, 0, 1, stride, range, cosine, sine);
    apply_gt_right(h, 0, 1, stride, range, cosine, sine);
    apply_g_right(t, 0, 1, stride, card, cosine, sine);
    for k in 1..range {
        let r = k * stride;
        let (_, cosine, sine) = implicit_givens_rotation(h[r + k], h[r + k + stride]);
        apply_g_left(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
        apply_gt_right(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
        apply_g_right(&mut t[r..], k, k + 1, stride, card, cosine, sine);
    }
}



fn main() {
    // test_reconstruct();
    // test_orthogonal();
}
