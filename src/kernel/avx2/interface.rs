use crate::arch::SIMD_WIDTH;
use crate::kernel::avx2::constants::{MASK, cfma_accum, fma_accum, mask_load, mask_store_ctrl};
use crate::kernel::avx2::{alligned, triangle, unalligned};
use std::arch::x86_64::{
    __m256, __m256i, _mm256_and_ps, _mm256_blendv_ps, _mm256_broadcast_ss, _mm256_castsi256_ps,
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskload_ps, _mm256_maskstore_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm256_stream_ps,
};
#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use stellar_macros::{kernel_mult_alligned, kernel_mult_unalligned};
// #[inline(always)]
// pub fn kernel_mult_simd(
//     mut xptr: *const f32,
//     mut yptr: *const f32,
//     tptr: *mut f32,
//     m: usize,
//     p: usize,
//     n: usize,
//     s_x: usize,
//     s_y: usize,
//     s_t: usize,
// ) {
//     // happens over k-contraction needs the imult kernel
//     unsafe {
//         if (m | n) & (SIMD_WIDTH - 1) == 0 {
//             kernel_mult_alligned!(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
//         } else {
//             kernel_mult_unalligned!(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
//         }
//     }
// }
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
