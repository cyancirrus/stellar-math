use crate::kernel::avx2::constants::{
    MASK, cfma_accum, mask_load, mask_load_ctrl, mask_store, mask_store_ctrl,
};
use std::arch::x86_64::{
    __m256i, _mm256_add_ps, _mm256_loadu_si256, _mm256_maskload_ps, _mm256_setzero_ps,
};
use stellar_macros::kernel_mult_unalligned;
#[target_feature(enable = "avx,avx2,fma")]
pub fn kernel_imult_safe(
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
    kernel_mult_unalligned!(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
}
#[target_feature(enable = "avx,avx2,fma")]
pub fn kernel_mult_safe(
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
    // w: workspace
    // excels at tall x matrix and wide y
    unsafe {
        let mask_p = MASK[p];
        let mask_n_ptr = MASK[n].as_ptr() as *const __m256i;
        let mask_n = _mm256_loadu_si256(mask_n_ptr);
        let row0 = mask_load_ctrl(mask_p[0], mask_n, yptr);
        let row4 = mask_load_ctrl(mask_p[4], mask_n, yptr.add(s_y * 4));
        let row1 = mask_load_ctrl(mask_p[1], mask_n, yptr.add(s_y));
        let row5 = mask_load_ctrl(mask_p[5], mask_n, yptr.add(s_y * 5));
        let row2 = mask_load_ctrl(mask_p[2], mask_n, yptr.add(s_y * 2));
        let row6 = mask_load_ctrl(mask_p[6], mask_n, yptr.add(s_y * 6));
        let row3 = mask_load_ctrl(mask_p[3], mask_n, yptr.add(s_y * 3));
        let row7 = mask_load_ctrl(mask_p[7], mask_n, yptr.add(s_y * 7));
        for _ in 0..m {
            let mut acc1 = _mm256_maskload_ps(tptr, mask_n);
            let mut acc0 = _mm256_setzero_ps();
            // _mm_prefetch(xptr.add(s_x) as *const i8, _MM_HINT_T0);
            // _mm_prefetch(tptr.add(s_t) as *const i8, _MM_HINT_T0);
            // start with existing t for accumulation
            acc0 = cfma_accum(mask_p[0], acc0, xptr, row0);
            acc1 = cfma_accum(mask_p[4], acc1, xptr.add(4), row4);
            acc0 = cfma_accum(mask_p[1], acc0, xptr.add(1), row1);
            acc1 = cfma_accum(mask_p[5], acc1, xptr.add(5), row5);
            acc0 = cfma_accum(mask_p[2], acc0, xptr.add(2), row2);
            acc1 = cfma_accum(mask_p[6], acc1, xptr.add(6), row6);
            acc0 = cfma_accum(mask_p[3], acc0, xptr.add(3), row3);
            acc1 = cfma_accum(mask_p[7], acc1, xptr.add(7), row7);
            mask_store(mask_n, tptr, _mm256_add_ps(acc1, acc0));
            xptr = xptr.add(s_x);
            tptr = tptr.add(s_t);
        }
    }
}
#[cfg(test)]
#[allow(dead_code, unused_imports, unused)]
mod test_safe_kernels {
    use super::*;
    use crate::algebra::bmethods::tensor_blockkern;
    use crate::algebra::ndmethods::basic_mult;
    use crate::arch::SIMD_WIDTH;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;
    #[cfg(feature = "avx2")]
    #[test]
    fn test_safe_kernels_dimensions() {
        for i in 1..=8 {
            for k in 1..=8 {
                for j in 1..=8 {
                    println!("i {i:}, k: {k:}, j: {j:}");
                    test_mpn_dimensions(i, k, j);
                }
            }
        }
    }
    #[cfg(feature = "avx2")]
    fn test_mpn_dimensions(m: usize, p: usize, n: usize) {
        unsafe {
            let (s_x, s_y, s_z) = (p, n, n);
            let mut x = generate_random_matrix(m, p);
            let mut y = generate_random_matrix(p, n);
            let expect = basic_mult(&x, &y);
            let mut x_simd = x.data.clone();
            let mut y_simd = y.data.clone();
            let mut t = vec![0f32; m * n];
            kernel_mult_safe(
                x_simd.as_ptr(),
                y_simd.as_ptr(),
                t.as_mut_ptr(),
                m,
                p,
                n,
                s_x,
                s_y,
                s_z,
            );
            // let inspect = NdArray {dims: vec![m, n], data: t.clone()};
            // println!("expected {expect:?}");
            // println!("actual {inspect:?}");
            assert!(approx_vector_eq(&expect.data, &t));
            let mut x_simd = x.data.clone();
            let mut y_simd = y.data.clone();
            let mut t = vec![0f32; m * n];
            kernel_imult_safe(
                x_simd.as_ptr(),
                y_simd.as_ptr(),
                t.as_mut_ptr(),
                m,
                p,
                n,
                s_x,
                s_y,
                s_z,
            );
            assert!(approx_vector_eq(&expect.data, &t));
        }
    }
}
