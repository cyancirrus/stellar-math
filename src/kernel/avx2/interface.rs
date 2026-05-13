#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use crate::arch::SIMD_WIDTH;
use crate::kernel::avx2::{alligned, triangle, unalligned};
#[inline(always)]
pub fn kernel_mult_simd(
    xptr: *const f32,
    yptr: *const f32,
    tptr: *mut f32,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // happens over k-contraction needs the imult kernel
    unsafe {
        if (m | n) & (SIMD_WIDTH - 1) == 0 {
            alligned::kernel_imult_simd_aligned(xptr, yptr, tptr, p, s_x, s_y, s_t);
        } else {
            unalligned::kernel_imult_safe(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
#[inline(always)]
pub fn kernel_lt_mult_simd(
    xptr: *const f32,
    yptr: *const f32,
    tptr: *mut f32,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        triangle::kernel_imult_lt_unalligned(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
    }
}
