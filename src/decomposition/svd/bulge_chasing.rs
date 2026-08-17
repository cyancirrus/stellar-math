#[rustfmt::skip]
use crate::decomposition::sgivens::{
    apply_g_left,
    apply_gt_right,
    implicit_givens_rotation,
};
use crate::decomposition::svd::primitives::{deflate, singular};
#[rustfmt::skip]
pub fn decomp_usym(
    b: &mut [f32],
    card: usize,
    stride: usize,
    max_iters:usize,
    tolerance: f32,
    absolute: f32,
) {

    let mut range = card;
    let mut inter = card.saturating_sub(2);
    let s = card * stride;
    // error 1 supra-diagonal above the first real eigen
    let mut e1 = s.saturating_sub(stride + 1);
    let mut tl = s.saturating_sub(stride + 2);
    let mut bl = s.saturating_sub(2);
    let mut curriter = 0;
    while range > 1 && curriter < max_iters {
        curriter += 1;
        let scale = b[tl].abs() + b[bl+1].abs();
        if b[e1].abs() < (scale * tolerance).min(absolute) {
            deflate(
                1,
                stride,
                &mut range,
                &mut inter,
                &mut e1,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
        } else {
            ugivens_iteration(b, inter, stride, tl, bl);
        }
    }
    if curriter == max_iters {
        println!("budget-exceeded");
    }
}

// A'A
// (a00, a10) * (a00, a01)
// (a01 a11)    (a10, a11)


// AA'
// (a00, a01) * (a00, a10)
// (a10, a11)   (a01, a11)

fn ugivens_iteration(h: &mut [f32], interior: usize, stride: usize, tl: usize, bl: usize) {
    let mut offset = 0;
    // push zero into col
    let sing = singular(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let sq_00 = h[0] * h[0];
    let sq_01 = h[0] * h[1];
    let (_, cos, sin) = implicit_givens_rotation(sq_00 - sing, sq_01);
    apply_gt_right(h, 0, 1, stride, 2, cos, sin);
    for _ in 0..interior {
        // push zero into row
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
        apply_g_left(&mut h[offset..], 0, 1, stride, 3, cos, sin);
        // push zero into col
        offset += 1;
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
        apply_gt_right(&mut h[offset..], 0, 1, stride, 3, cos, sin);
        offset += stride;
    }
    // push zero into row
    let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
    apply_g_left(&mut h[offset..], 0, 1, stride, 2, cos, sin);
}


#[rustfmt::skip]
pub fn decomp_lsym(
    b: &mut [f32],
    card: usize,
    stride: usize,
    max_iters:usize,
    tolerance: f32,
    absolute: f32,
) {

    let mut range = card;
    let mut inter = card.saturating_sub(2);
    let s = card * stride;
    // error 1 supra-diagonal above the first real eigen
    let mut tl = (s + card).saturating_sub(2 + stride * 2);
    let mut bl = (s + card).saturating_sub(stride + 2);
    let mut e1 = (s + card).saturating_sub(stride + 2);
    let mut curriter = 0;
    while range > 1 && curriter < max_iters {
        curriter += 1;
        let scale = b[tl].abs() + b[bl+1].abs();
        if b[e1].abs() < (scale * tolerance).min(absolute) {
            println!("deflating");
            deflate(
                1,
                stride,
                &mut range,
                &mut inter,
                &mut e1,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
        } else {
            lgivens_iteration(b, inter, stride, tl, bl);
        }
    }
    if curriter == max_iters {
        println!("budget-exceeded");
    }
}

#[rustfmt::skip]
fn lgivens_iteration(h: &mut [f32], interior: usize, stride: usize, tl: usize, bl: usize) {
    let mut offset = 0;
    // push zero into row
    let sing = singular(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let sq_00 = h[0] * h[0];
    let sq_10 = h[0] * h[stride];
    // let (_, cos, sin) = implicit_givens_rotation(h[0], h[stride]);
    let (_, cos, sin) = implicit_givens_rotation(sq_00 - sing, sq_10);
    apply_g_left(h, 0, 1, stride, 2, cos, sin);
    for _ in 0..interior {
        // push zero into col
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
        apply_gt_right(&mut h[offset ..], 0, 1, stride, 3, cos, sin);
        // push zero into row
        offset += stride;
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
        apply_g_left(&mut h[offset..], 0, 1, stride, 3, cos, sin);
        offset += 1;
    }
    // // push zero into col
    let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
    apply_gt_right(&mut h[offset ..], 0, 1, stride, 2, cos, sin);
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
















fn iteration_ugivens(h: &mut [f32], interior: usize, stride: usize) -> f32 {
    let mut supdiag_norm = 0f32;
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
        apply_gt_right(&mut h[offset..], 0, 1, stride, 3, cos, sin);
        supdiag_norm += h[offset].abs();
        offset += stride;
    }
    // push zero into row
    let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
    apply_g_left(&mut h[offset..], 0, 1, stride, 2, cos, sin);
    supdiag_norm + h[offset + 1].abs()
}
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
        supdiag_norm = iteration_ugivens(h, interior, stride);
    }
}
