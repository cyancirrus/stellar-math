use crate::arch::SIMD_WIDTH;
use crate::kernel::avx2::constants::{MASK, cfma_accum, fma_accum, mask_load, mask_store_ctrl};
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_broadcast_ss, _mm256_castpd_ps, _mm256_castps_pd, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_permute2f128_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm256_unpackhi_pd, _mm256_unpackhi_ps, _mm256_unpacklo_pd, _mm256_unpacklo_ps,
};
use stellar_macros::{kernel_mult_alligned, kernel_mult_unalligned};
macro_rules! fma_accum {
    ($acc:expr, $cptr:expr, $data:expr) => {
        $acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&*$cptr), $data, $acc);
    };
}
#[target_feature(enable = "avx,avx2,fma")]
pub fn kernel_imult_simd_aligned(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    p: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    kernel_mult_alligned!(xptr, yptr, tptr, SIMD_WIDTH, p, SIMD_WIDTH, s_x, s_y, s_t);
}
#[target_feature(enable = "avx,avx2,fma")]
pub fn kernel_mult_simd_aligned(
    mut xptr: *const f32,
    yptr: *const f32,
    mut tptr: *mut f32,
    m: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // excels at tall x matrix and wide y
    unsafe {
        let row0 = _mm256_loadu_ps(yptr);
        let row4 = _mm256_loadu_ps(yptr.add(s_y * 4));
        let row1 = _mm256_loadu_ps(yptr.add(s_y));
        let row5 = _mm256_loadu_ps(yptr.add(s_y * 5));
        let row2 = _mm256_loadu_ps(yptr.add(s_y * 2));
        let row6 = _mm256_loadu_ps(yptr.add(s_y * 6));
        let row3 = _mm256_loadu_ps(yptr.add(s_y * 3));
        let row7 = _mm256_loadu_ps(yptr.add(s_y * 7));
        // t is being passed in
        for _ in 0..m {
            let mut acc1 = _mm256_loadu_ps(tptr);
            let mut acc0 = _mm256_setzero_ps();
            // _mm_prefetch(xptr.add(s_x) as *const i8, _MM_HINT_T0);
            // _mm_prefetch(tptr.add(s_t) as *const i8, _MM_HINT_T0);
            // start with existing t for accumulation
            fma_accum!(acc0, xptr, row0);
            fma_accum!(acc1, xptr.add(4), row4);
            fma_accum!(acc0, xptr.add(1), row1);
            fma_accum!(acc1, xptr.add(5), row5);
            fma_accum!(acc0, xptr.add(2), row2);
            fma_accum!(acc1, xptr.add(6), row6);
            fma_accum!(acc0, xptr.add(3), row3);
            fma_accum!(acc1, xptr.add(7), row7);
            _mm256_storeu_ps(tptr, _mm256_add_ps(acc1, acc0));
            xptr = xptr.add(s_x);
            tptr = tptr.add(s_t);
        }
    }
}
#[rustfmt::skip]
#[target_feature(enable = "avx,fma")]
// pub fn kernel_trans_simd(mut tptr: *mut f32, mut wptr: *mut f32) {
pub fn kernel_trans_simd(tptr: *mut f32) {
    unsafe {
        let r0 = _mm256_loadu_ps(tptr);
        let r1 = _mm256_loadu_ps(tptr.add(8));
        let r2 = _mm256_loadu_ps(tptr.add(8 * 2));
        let r3 = _mm256_loadu_ps(tptr.add(8 * 3));
        let r4 = _mm256_loadu_ps(tptr.add(8 * 4));
        let r5 = _mm256_loadu_ps(tptr.add(8 * 5));
        let r6 = _mm256_loadu_ps(tptr.add(8 * 6));
        let r7 = _mm256_loadu_ps(tptr.add(8 * 7));

        let t0 = _mm256_unpacklo_ps(r0, r1);
        let t1 = _mm256_unpackhi_ps(r0, r1);
        let t2 = _mm256_unpacklo_ps(r2, r3);
        let t3 = _mm256_unpackhi_ps(r2, r3);
        let t4 = _mm256_unpacklo_ps(r4, r5);
        let t5 = _mm256_unpackhi_ps(r4, r5);
        let t6 = _mm256_unpacklo_ps(r6, r7);
        let t7 = _mm256_unpackhi_ps(r6, r7);

        let q0 = _mm256_castpd_ps(_mm256_unpacklo_pd( _mm256_castps_pd(t0), _mm256_castps_pd(t2),));
        let q1 = _mm256_castpd_ps(_mm256_unpackhi_pd( _mm256_castps_pd(t0), _mm256_castps_pd(t2),));
        let q2 = _mm256_castpd_ps(_mm256_unpacklo_pd( _mm256_castps_pd(t1), _mm256_castps_pd(t3),));
        let q3 = _mm256_castpd_ps(_mm256_unpackhi_pd( _mm256_castps_pd(t1), _mm256_castps_pd(t3),));
        let q4 = _mm256_castpd_ps(_mm256_unpacklo_pd( _mm256_castps_pd(t4), _mm256_castps_pd(t6),));
        let q5 = _mm256_castpd_ps(_mm256_unpackhi_pd( _mm256_castps_pd(t4), _mm256_castps_pd(t6),));
        let q6 = _mm256_castpd_ps(_mm256_unpacklo_pd( _mm256_castps_pd(t5), _mm256_castps_pd(t7),));
        let q7 = _mm256_castpd_ps(_mm256_unpackhi_pd( _mm256_castps_pd(t5), _mm256_castps_pd(t7),));
        
        // 0 low of arg 1, 1 high of arg 1, 2 low of arg 2, 3 high of arg 2
        _mm256_storeu_ps(tptr, _mm256_permute2f128_ps(q0, q4, 0x20));
        _mm256_storeu_ps(tptr.add(8), _mm256_permute2f128_ps(q0, q4, 0x31));
        _mm256_storeu_ps(tptr.add(8 * 2), _mm256_permute2f128_ps(q1, q5, 0x20));
        _mm256_storeu_ps(tptr.add(8 * 3), _mm256_permute2f128_ps(q1, q5, 0x31));
        _mm256_storeu_ps(tptr.add(8 * 4), _mm256_permute2f128_ps(q2, q6, 0x20));
        _mm256_storeu_ps(tptr.add(8 * 5), _mm256_permute2f128_ps(q2, q6, 0x31));
        _mm256_storeu_ps(tptr.add(8 * 6), _mm256_permute2f128_ps(q3, q7, 0x20));
        _mm256_storeu_ps(tptr.add(8 * 7), _mm256_permute2f128_ps(q3, q7, 0x31));
    }
}
#[cfg(test)]
#[cfg(feature = "avx2")]
mod test_avx2_kernels {
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;
    const BLOCK_AVX2: usize = 8;
    #[test]
    fn test_kernel_8x8_kernels() {
        unsafe {
            let (m, p, n) = (8, 8, 8);
            let (s_x, s_y, s_z) = (8, 8, 8);
            let mut x = generate_random_matrix(m, p);
            let mut y = generate_random_matrix(p, n);
            let expect = basic_mult(&x, &y);
            let mut x_simd = x.data.clone();
            let mut y_simd = y.data.clone();
            let mut w = vec![0f32; 8 * 8];
            let mut t = vec![0f32; m * n];
            kernel_imult_simd_aligned(
                x_simd.as_ptr(),
                y_simd.as_ptr(),
                t.as_mut_ptr(),
                m,
                s_x,
                s_y,
                s_z,
            );
            assert!(approx_vector_eq(&expect.data, &t));
            let mut x_simd = x.data.clone();
            let mut y_simd = y.data.clone();
            let mut w = vec![0f32; 8 * 8];
            let mut t = vec![0f32; m * n];
            kernel_mult_simd_aligned(
                x_simd.as_ptr(),
                y_simd.as_ptr(),
                t.as_mut_ptr(),
                m,
                s_x,
                s_y,
                s_z,
            );
            assert!(approx_vector_eq(&expect.data, &t));
        }
    }
}
