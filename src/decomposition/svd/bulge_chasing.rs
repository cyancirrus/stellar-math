#[rustfmt::skip]
use crate::decomposition::sgivens::{
    apply_g_left,
    apply_gt_right,
    implicit_givens_rotation,
};
#[rustfmt::skip]
pub fn decomp_ugivens(
    h: &mut [f32],
    card: usize,
    stride: usize,
    max_iters:usize,
    threshold: f32,
) {
    let interior = card.saturating_sub(2);
    let mut supdiag_norm = f32::INFINITY;
    for _ in 0..max_iters {
        if supdiag_norm < threshold { break; }
        supdiag_norm = 0f32;
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
