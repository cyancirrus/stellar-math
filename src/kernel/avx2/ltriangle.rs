use crate::kernel::avx2::constants::{MASK, cfma_accum, mask_load, mask_store_ctrl};
use std::arch::x86_64::{__m256i, _mm256_loadu_si256};
#[target_feature(enable = "avx,avx2,fma")]
pub fn lmult_lt(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    pre: usize,
    pro: usize,
    pos: usize,
    m: usize,
    _p: usize,
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
        let mut row0 = mask_load(mask_n, tptr);
        let mut row1 = mask_load(mask_n, tptr.add(s_t));
        let mut row2 = mask_load(mask_n, tptr.add(s_t * 2));
        let mut row3 = mask_load(mask_n, tptr.add(s_t * 3));
        let mut row4 = mask_load(mask_n, tptr.add(s_t * 4));
        let mut row5 = mask_load(mask_n, tptr.add(s_t * 5));
        let mut row6 = mask_load(mask_n, tptr.add(s_t * 6));
        let mut row7 = mask_load(mask_n, tptr.add(s_t * 7));
        let mut mask_t = MASK[m];
        for idx in 0..pre {
            mask_t[idx] = 0;
        }
        for _k in 0..pro {
            let b0 = mask_load(mask_n, yptr);
            yptr = yptr.add(s_y);
            row0 = cfma_accum(mask_t[0], row0, xptr, b0);
            row1 = cfma_accum(mask_t[1], row1, xptr.add(s_x), b0);
            row2 = cfma_accum(mask_t[2], row2, xptr.add(2 * s_x), b0);
            row3 = cfma_accum(mask_t[3], row3, xptr.add(3 * s_x), b0);
            row4 = cfma_accum(mask_t[4], row4, xptr.add(4 * s_x), b0);
            row5 = cfma_accum(mask_t[5], row5, xptr.add(5 * s_x), b0);
            row6 = cfma_accum(mask_t[6], row6, xptr.add(6 * s_x), b0);
            row7 = cfma_accum(mask_t[7], row7, xptr.add(7 * s_x), b0);
            xptr = xptr.add(1);
        }
        // exit early for disappearing contractions
        let mask_m = mask_t;
        for k in 0..pos {
            mask_t[k + pre] = 0;
            let b0 = mask_load(mask_n, yptr);
            yptr = yptr.add(s_y);
            row0 = cfma_accum(mask_t[0], row0, xptr, b0);
            row1 = cfma_accum(mask_t[1], row1, xptr.add(s_x), b0);
            row2 = cfma_accum(mask_t[2], row2, xptr.add(2 * s_x), b0);
            row3 = cfma_accum(mask_t[3], row3, xptr.add(3 * s_x), b0);
            row4 = cfma_accum(mask_t[4], row4, xptr.add(4 * s_x), b0);
            row5 = cfma_accum(mask_t[5], row5, xptr.add(5 * s_x), b0);
            row6 = cfma_accum(mask_t[6], row6, xptr.add(6 * s_x), b0);
            row7 = cfma_accum(mask_t[7], row7, xptr.add(7 * s_x), b0);
            xptr = xptr.add(1);
        }
        mask_store_ctrl(mask_m[0], mask_n, tptr, row0);
        mask_store_ctrl(mask_m[1], mask_n, tptr.add(s_t), row1);
        mask_store_ctrl(mask_m[2], mask_n, tptr.add(s_t * 2), row2);
        mask_store_ctrl(mask_m[3], mask_n, tptr.add(s_t * 3), row3);
        mask_store_ctrl(mask_m[4], mask_n, tptr.add(s_t * 4), row4);
        mask_store_ctrl(mask_m[5], mask_n, tptr.add(s_t * 5), row5);
        mask_store_ctrl(mask_m[6], mask_n, tptr.add(s_t * 6), row6);
        mask_store_ctrl(mask_m[7], mask_n, tptr.add(s_t * 7), row7);
    }
}
#[target_feature(enable = "avx,avx2,fma")]
pub fn lmult_ut(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    pre: usize,
    pro: usize,
    pos: usize,
    m: usize,
    _p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // debug_assert!(pre <= m);
    // Sum[K] Union[I] { g^i = aik b^k }
    // excels at processing panels of data ie 8 x K * K x 8;
    unsafe {
        let mask_n_ptr = MASK[n].as_ptr() as *const __m256i;
        let mask_n = _mm256_loadu_si256(mask_n_ptr);
        let mut row0 = mask_load(mask_n, tptr);
        let mut row1 = mask_load(mask_n, tptr.add(s_t));
        let mut row2 = mask_load(mask_n, tptr.add(s_t * 2));
        let mut row3 = mask_load(mask_n, tptr.add(s_t * 3));
        let mut row4 = mask_load(mask_n, tptr.add(s_t * 4));
        let mut row5 = mask_load(mask_n, tptr.add(s_t * 5));
        let mut row6 = mask_load(mask_n, tptr.add(s_t * 6));
        let mut row7 = mask_load(mask_n, tptr.add(s_t * 7));
        let mut mask_t = MASK[0];
        for idx in 0..pre {
            mask_t[idx] = -1;
        }
        // handle the triangle portion
        for k in 0..pos {
            mask_t[k + pre] = -1;
            let b0 = mask_load(mask_n, yptr);
            yptr = yptr.add(s_y);
            row0 = cfma_accum(mask_t[0], row0, xptr, b0);
            row1 = cfma_accum(mask_t[1], row1, xptr.add(s_x), b0);
            row2 = cfma_accum(mask_t[2], row2, xptr.add(2 * s_x), b0);
            row3 = cfma_accum(mask_t[3], row3, xptr.add(3 * s_x), b0);
            row4 = cfma_accum(mask_t[4], row4, xptr.add(4 * s_x), b0);
            row5 = cfma_accum(mask_t[5], row5, xptr.add(5 * s_x), b0);
            row6 = cfma_accum(mask_t[6], row6, xptr.add(6 * s_x), b0);
            row7 = cfma_accum(mask_t[7], row7, xptr.add(7 * s_x), b0);
            xptr = xptr.add(1);
        }
        for _k in 0..pro {
            let b0 = mask_load(mask_n, yptr);
            yptr = yptr.add(s_y);
            row0 = cfma_accum(mask_t[0], row0, xptr, b0);
            row1 = cfma_accum(mask_t[1], row1, xptr.add(s_x), b0);
            row2 = cfma_accum(mask_t[2], row2, xptr.add(2 * s_x), b0);
            row3 = cfma_accum(mask_t[3], row3, xptr.add(3 * s_x), b0);
            row4 = cfma_accum(mask_t[4], row4, xptr.add(4 * s_x), b0);
            row5 = cfma_accum(mask_t[5], row5, xptr.add(5 * s_x), b0);
            row6 = cfma_accum(mask_t[6], row6, xptr.add(6 * s_x), b0);
            row7 = cfma_accum(mask_t[7], row7, xptr.add(7 * s_x), b0);
            xptr = xptr.add(1);
        }
        mask_store_ctrl(mask_t[0], mask_n, tptr, row0);
        mask_store_ctrl(mask_t[1], mask_n, tptr.add(s_t), row1);
        mask_store_ctrl(mask_t[2], mask_n, tptr.add(s_t * 2), row2);
        mask_store_ctrl(mask_t[3], mask_n, tptr.add(s_t * 3), row3);
        mask_store_ctrl(mask_t[4], mask_n, tptr.add(s_t * 4), row4);
        mask_store_ctrl(mask_t[5], mask_n, tptr.add(s_t * 5), row5);
        mask_store_ctrl(mask_t[6], mask_n, tptr.add(s_t * 6), row6);
        mask_store_ctrl(mask_t[7], mask_n, tptr.add(s_t * 7), row7);
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
}
