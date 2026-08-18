use crate::decomposition::svd::constants::MAX_ITERS;

pub fn singular(m00: f32, m01: f32, m10: f32, m11: f32) -> f32 {
    let off_diag = m00 * m01 + m10 * m11;
    let s00 = m00 * m00 + m10 * m10;
    let s11 = m01 * m01 + m11 * m11;
    let d = (s00 - s11) / 2f32;
    let discriminate = d * d + off_diag * off_diag;
    s11 + d - d.signum() * discriminate.max(0f32).sqrt()
}
// element in the m10 is zero
pub fn upper_singular(sq_00: f32, sq_01: f32, sq_11: f32) -> f32 {
    // let off_diag = m00 * m01;
    // let s00 = m00 * m00;
    // let s11 = m01 * m01 + m11 * m11;
    let d = (sq_00 - sq_11) / 2f32;
    let discriminate = d * d + sq_01 * sq_01;
    sq_11 + d - d.signum() * discriminate.max(0f32).sqrt()
}

pub fn deflate(
    amount: usize,
    stride: usize,
    range: &mut usize,
    inter: &mut usize,
    e1: &mut usize,
    tl: &mut usize,
    bl: &mut usize,
    // stall: &mut usize,
    curriter: &mut usize,
) {
    let shift = amount * stride + amount;
    *range -= amount;
    *inter = inter.saturating_sub(amount);
    *e1 = e1.saturating_sub(shift);
    *tl = tl.saturating_sub(shift);
    *bl = bl.saturating_sub(shift);
    *curriter = curriter.saturating_sub(MAX_ITERS >> 1);
}
