use crate::decomposition::francis::constants::{MAX_ITERS, TOLERANCE};
use crate::decomposition::sgivens::{apply_g_right, apply_gt_left, implicit_givens_rotation};
#[rustfmt::skip]
use crate::decomposition::francis::primitives::{
    deflate,
    eigen,
};
use crate::structure::ndarray::NdArray;

pub fn decomp_sym(h: &mut [f32], mut range: usize, size: usize, stride: usize) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(stride + 1);
    let mut e2 = s.saturating_sub(stride + stride + 2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut bl = s.saturating_sub(2);
    let mut curriter = 0;
    let _he1 = h[e1];
    let _he2 = h[e2];
    while range > 1 && curriter < MAX_ITERS {
        curriter += 1;
        if h[e1].abs() < TOLERANCE {
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
        } else {
            francis_iteration_sym(h, size, range, stride, tl, bl);
        }
    }
}
/// francis_iteration_sym
///
/// * h: hessenberg linearized matrix
/// * size: static number of rows for rotations
/// * range: number of rows in active window
/// * stride: stride of the data format
/// * tl: top left of the window for the eigens
/// * bl: bottom left of the window for the eigens
pub fn francis_iteration_sym(
    h: &mut [f32],
    _size: usize,
    range: usize,
    stride: usize,
    tl: usize,
    bl: usize,
) {
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[1]);
    apply_g_right(h, 0, 1, stride, range, cosine, -sine);
    apply_gt_left(h, 0, 1, stride, range, cosine, -sine);
    for o in 0..range.saturating_sub(2) {
        let r = o * stride;
        let s1 = o + 1;
        let s2 = o + 2;
        let _temp = NdArray {
            dims: vec![range, range],
            data: h.to_vec(),
        };
        let (_, cosine, sine) = implicit_givens_rotation(h[r + s1], h[r + s2]);
        // apply_g_right(&mut h[r..], s1, s2, stride, size - o, cosine, -sine);
        // apply_gt_left(h, s1, s2, stride, range.min(s2 + 2), cosine, -sine);
        apply_g_right(&mut h[r..], s1, s2, stride, range - o, cosine, -sine);
        apply_gt_left(h, s1, s2, stride, range, cosine, -sine);
    }
}
