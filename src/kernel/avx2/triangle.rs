use crate::kernel::avx2::constants::{MASK, mask_load, mask_store_ctrl};
use std::arch::x86_64::{
    __m256, __m256i, _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_and_ps, _mm256_broadcast_ss,
    _mm256_castsi256_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskstore_ps,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};
macro_rules! fma_gated {
    ($acc:expr, $ptr:expr, $mask_bit:expr, $data:expr) => {
        if $mask_bit != 0 {
            $acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&*$ptr), $data, $acc);
            // $acc = _mm256_fmadd_ps(_mm256_set1_ps(*$ptr), $b, $acc);
        }
    };
}
#[target_feature(enable = "avx,avx2,fma")]
pub fn kernel_imult_lt_unalligned(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // Sum[K] Union[I] { g^i = aik b^k }
    // excels at processing panels of data ie 8 x K * K x 8;
    unsafe {
        let mask_n_ptr = MASK[n].as_ptr() as *const __m256i;
        let mask_n = _mm256_loadu_si256(mask_n_ptr);
        let mut i_row = mask_load(tptr, mask_n);
        let mut v_row = mask_load(tptr.add(s_t * 4), mask_n);
        let mut ii_row = mask_load(tptr.add(s_t), mask_n);
        let mut vi_row = mask_load(tptr.add(s_t * 5), mask_n);
        let mut iii_row = mask_load(tptr.add(s_t * 2), mask_n);
        let mut vii_row = mask_load(tptr.add(s_t * 6), mask_n);
        let mut iv_row = mask_load(tptr.add(s_t * 3), mask_n);
        let mut viii_row = mask_load(tptr.add(s_t * 7), mask_n);
        let threshold = m.min(p);
        let mask_m = MASK[m];
        for k in threshold..p {
            // _mm_prefetch(yptr.add(s_y) as *const i8, _MM_HINT_T0);
            // _mm_prefetch(xptr.add(4 * s_x) as *const i8, _MM_HINT_T0);
            let b0 = mask_load(yptr, mask_n);
            yptr = yptr.add(s_y);
            fma_gated!(i_row, xptr, mask_m[0], b0);
            fma_gated!(ii_row, xptr.add(s_x), mask_m[1], b0);
            fma_gated!(iii_row, xptr.add(2 * s_x), mask_m[2], b0);
            fma_gated!(iv_row, xptr.add(3 * s_x), mask_m[3], b0);
            fma_gated!(v_row, xptr.add(4 * s_x), mask_m[4], b0);
            fma_gated!(vi_row, xptr.add(5 * s_x), mask_m[5], b0);
            fma_gated!(vii_row, xptr.add(6 * s_x), mask_m[6], b0);
            fma_gated!(viii_row, xptr.add(7 * s_x), mask_m[7], b0);
            xptr = xptr.add(1);
        }
        let mut mask_t = mask_m;
        for k in 0..threshold {
            let b0 = mask_load(yptr, mask_n);
            yptr = yptr.add(s_y);
            fma_gated!(i_row, xptr, mask_t[0], b0);
            fma_gated!(ii_row, xptr.add(s_x), mask_t[1], b0);
            fma_gated!(iii_row, xptr.add(2 * s_x), mask_t[2], b0);
            fma_gated!(iv_row, xptr.add(3 * s_x), mask_t[3], b0);
            fma_gated!(v_row, xptr.add(4 * s_x), mask_t[4], b0);
            fma_gated!(vi_row, xptr.add(5 * s_x), mask_t[5], b0);
            fma_gated!(vii_row, xptr.add(6 * s_x), mask_t[6], b0);
            fma_gated!(viii_row, xptr.add(7 * s_x), mask_t[7], b0);
            mask_t[k] = 0;
            xptr = xptr.add(1);
        }
        mask_store_ctrl(tptr, mask_n, i_row, mask_m[0]);
        mask_store_ctrl(tptr.add(s_t * 4), mask_n, v_row, mask_m[4]);
        mask_store_ctrl(tptr.add(s_t), mask_n, ii_row, mask_m[1]);
        mask_store_ctrl(tptr.add(s_t * 5), mask_n, vi_row, mask_m[5]);
        mask_store_ctrl(tptr.add(s_t * 2), mask_n, iii_row, mask_m[2]);
        mask_store_ctrl(tptr.add(s_t * 6), mask_n, vii_row, mask_m[6]);
        mask_store_ctrl(tptr.add(s_t * 3), mask_n, iv_row, mask_m[3]);
        mask_store_ctrl(tptr.add(s_t * 7), mask_n, viii_row, mask_m[7]);
    }
}
#[target_feature(enable = "avx,avx2,fma")]
pub fn kernel_imult_ut_unalligned(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // Sum[K] Union[I] { g^i = aik b^k }
    // excels at processing panels of data ie 8 x K * K x 8;
    unsafe {
        let mask_n_ptr = MASK[n].as_ptr() as *const __m256i;
        let mask_n = _mm256_loadu_si256(mask_n_ptr);
        let mut i_row = mask_load(tptr, mask_n);
        let mut v_row = mask_load(tptr.add(s_t * 4), mask_n);
        let mut ii_row = mask_load(tptr.add(s_t), mask_n);
        let mut vi_row = mask_load(tptr.add(s_t * 5), mask_n);
        let mut iii_row = mask_load(tptr.add(s_t * 2), mask_n);
        let mut vii_row = mask_load(tptr.add(s_t * 6), mask_n);
        let mut iv_row = mask_load(tptr.add(s_t * 3), mask_n);
        let mut viii_row = mask_load(tptr.add(s_t * 7), mask_n);
        let mask_m = MASK[m];
        let mut mask_t = MASK[0];
        let threshold = m.min(p);
        for k in 0..threshold {
            mask_t[k] = mask_m[k];
            let b0 = mask_load(yptr, mask_n);
            yptr = yptr.add(s_y);
            fma_gated!(i_row, xptr, mask_t[0], b0);
            fma_gated!(ii_row, xptr.add(s_x), mask_t[1], b0);
            fma_gated!(iii_row, xptr.add(2 * s_x), mask_t[2], b0);
            fma_gated!(iv_row, xptr.add(3 * s_x), mask_t[3], b0);
            fma_gated!(v_row, xptr.add(4 * s_x), mask_t[4], b0);
            fma_gated!(vi_row, xptr.add(5 * s_x), mask_t[5], b0);
            fma_gated!(vii_row, xptr.add(6 * s_x), mask_t[6], b0);
            fma_gated!(viii_row, xptr.add(7 * s_x), mask_t[7], b0);
            xptr = xptr.add(1);
        }
        for k in threshold..p {
            // _mm_prefetch(yptr.add(s_y) as *const i8, _MM_HINT_T0);
            // _mm_prefetch(xptr.add(4 * s_x) as *const i8, _MM_HINT_T0);
            let b0 = mask_load(yptr, mask_n);
            yptr = yptr.add(s_y);
            fma_gated!(i_row, xptr, mask_m[0], b0);
            fma_gated!(ii_row, xptr.add(s_x), mask_m[1], b0);
            fma_gated!(iii_row, xptr.add(2 * s_x), mask_m[2], b0);
            fma_gated!(iv_row, xptr.add(3 * s_x), mask_m[3], b0);
            fma_gated!(v_row, xptr.add(4 * s_x), mask_m[4], b0);
            fma_gated!(vi_row, xptr.add(5 * s_x), mask_m[5], b0);
            fma_gated!(vii_row, xptr.add(6 * s_x), mask_m[6], b0);
            fma_gated!(viii_row, xptr.add(7 * s_x), mask_m[7], b0);
            xptr = xptr.add(1);
        }
        mask_store_ctrl(tptr, mask_n, i_row, mask_m[0]);
        mask_store_ctrl(tptr.add(s_t * 4), mask_n, v_row, mask_m[4]);
        mask_store_ctrl(tptr.add(s_t), mask_n, ii_row, mask_m[1]);
        mask_store_ctrl(tptr.add(s_t * 5), mask_n, vi_row, mask_m[5]);
        mask_store_ctrl(tptr.add(s_t * 2), mask_n, iii_row, mask_m[2]);
        mask_store_ctrl(tptr.add(s_t * 6), mask_n, vii_row, mask_m[6]);
        mask_store_ctrl(tptr.add(s_t * 3), mask_n, iv_row, mask_m[3]);
        mask_store_ctrl(tptr.add(s_t * 7), mask_n, viii_row, mask_m[7]);
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
    fn test_triangle_equalities() {
        let (m, k, n) = (8, 8, 8);
        let mut output = vec![0f32; m * n];
        test_lower_triangle(m, k, n, &mut output);
        output.fill(0f32);
        test_upper_triangle(m, k, n, &mut output);
        output.fill(0f32);
    }
    fn filter_lower_triangle(a: &mut NdArray) {
        let (rows, cols) = (a.dims[0], a.dims[1]);
        let d = &mut a.data;
        for i in 0..rows {
            for j in i + 1..cols {
                d[i * cols + j] = 0f32;
            }
        }
    }
    fn filter_upper_triangle(a: &mut NdArray) {
        let (rows, cols) = (a.dims[0], a.dims[1]);
        let d = &mut a.data;
        for i in 1..rows {
            for j in 0..i {
                d[i * cols + j] = 0f32;
            }
        }
    }
    fn test_lower_triangle(m: usize, k: usize, n: usize, output: &mut [f32]) {
        let a = generate_random_matrix(m, k);
        let mut a_control = a.clone();
        filter_lower_triangle(&mut a_control);
        let b = generate_random_matrix(k, n);
        let expected = basic_mult(&a_control, &b);
        unsafe {
            kernel_imult_lt_unalligned(
                a.data.as_ptr(),
                b.data.as_ptr(),
                output.as_mut_ptr(),
                BLOCK_AVX2,
                BLOCK_AVX2,
                BLOCK_AVX2,
                m,
                n,
                n,
            );
        }
        debug_assert!(approx_vector_eq(&expected.data, &output[..m * n]));
    }
    fn test_upper_triangle(m: usize, k: usize, n: usize, output: &mut [f32]) {
        let a = generate_random_matrix(m, k);
        let mut a_control = a.clone();
        let b = generate_random_matrix(k, n);
        filter_upper_triangle(&mut a_control);
        let expected = basic_mult(&a_control, &b);
        unsafe {
            kernel_imult_ut_unalligned(
                a.data.as_ptr(),
                b.data.as_ptr(),
                output.as_mut_ptr(),
                BLOCK_AVX2,
                BLOCK_AVX2,
                BLOCK_AVX2,
                m,
                n,
                n,
            );
        }
        debug_assert!(approx_vector_eq(&expected.data, &output[..m * n]));
    }
}
