#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use crate::arch::SIMD_WIDTH;
use crate::kernel::avx2::{alligned, unalligned};
use crate::kernel::default::kernel_mult_scalar;
use std::arch::x86_64::{
    _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_broadcast_ss, _mm256_castpd_ps,
    _mm256_castps_pd, _mm256_fmadd_ps, _mm256_load_ps, _mm256_loadu_ps, _mm256_mask_load_ps,
    _mm256_permute2f128_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm256_unpackhi_pd, _mm256_unpackhi_ps, _mm256_unpacklo_pd, _mm256_unpacklo_ps,
};
#[inline(always)]
pub fn kernel_mult_simd(
    mut xptr: *const f32,
    yptr: *const f32,
    mut tptr: *mut f32,
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
