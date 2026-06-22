use crate::arch::SIMD_WIDTH;
use crate::kernel::matkerns::kernel_mult;
use rayon::prelude::*;


/// # pack transfers a copy of data from d to pack
/// * to inverse simply exchange d and b
/// - d ~ M(r, s)
///
/// * d: contains the source data of x sliced to begin at mc
/// * b: contains the target pack for the outer iteration loop
/// * re: size of the r-block
/// * se: size of the s-block
/// * s_b: stride of block
/// * s_d: stride of the matrix d
#[inline(always)]
pub fn pack(d: &[f32], b: &mut [f32], re: usize, se: usize, s_b: usize, s_d: usize) {
    unsafe {
        let mut doffset = 0;
        let mut boffset = 0;
        for _ in 0..re {
            b.get_unchecked_mut(boffset..boffset + se)
                .copy_from_slice(&d.get_unchecked(doffset..doffset + se));
            boffset += s_b;
            doffset += s_d;
        }
    }
}
/// (x - b).min(t)
#[inline(always)]
pub fn diff_min(x: usize, b: usize, t: usize) -> usize {
    if x - b < t { x - b } else { t }
}
pub fn tensor_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut xoffset = 0;
        let mut toffset = 0;
        let dx = SIMD_WIDTH * s_x;
        let dt = SIMD_WIDTH * s_t;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            for j in (0..n).step_by(SIMD_WIDTH) {
                let jj_end = SIMD_WIDTH.min(n - j);
                kernel_mult(
                    x_d.get_unchecked(xoffset..),
                    y_d.get_unchecked(j..),
                    t_d.get_unchecked_mut(toffset + j..),
                    ii_end,
                    p,
                    jj_end,
                    s_x,
                    s_y,
                    s_t,
                );
            }
            toffset += dt;
            xoffset += dx;
        }
    }
}
