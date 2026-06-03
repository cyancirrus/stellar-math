use crate::kernel::avx2::constants::{MASK, cfma_accum, mask_load, mask_store_ctrl};
use std::arch::x86_64::{__m256i, _mm256_loadu_si256};

// d : diagonal
pub fn lmult_lt_tail(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    mut d: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    //println!("p: {p:}, d {d:?}");
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
        for idx in 0..d {
            mask_t[idx] = 0;
        }
        //println!("row7 {row7:?}");
        let mut mask_m = mask_t;
        // mask_m[d] = 0;
        //println!("mask_m {mask_m:?}");
        // for k in d..p {
        // for k in d..p  {
        for k in d..p + d  {
            mask_t[k] = 0;
            //println!("tail mask_t {mask_t:?}");
            // //println!("tail");
            let b0 = mask_load(mask_n, yptr);
            //println!("b0 {b0:?}");
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
        // //println!("row7 {row7:?}");
    }
}

#[target_feature(enable = "avx,avx2,fma")]
pub fn lmult_lt_tri(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    d: usize,
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
        //println!("what is p {p:?}, d {d:?}");
        // //println!("s_x {s_x:}, s_y: {s_y:}, s_t: {s_t:}");

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
        let threshold = m.min(p);
        // //println!("m {m:}, p: {p:}, n: {n:}");
        //println!("mask_n {:?}", MASK[n]);
        let mask_m = MASK[m];
        for _k in 0..d  {
            let b0 = mask_load(mask_n, yptr);
            // //println!("b0 {b0:?}");
            yptr = yptr.add(s_y);
            row0 = cfma_accum(mask_m[0], row0, xptr, b0);
            row1 = cfma_accum(mask_m[1], row1, xptr.add(s_x), b0);
            row2 = cfma_accum(mask_m[2], row2, xptr.add(2 * s_x), b0);
            row3 = cfma_accum(mask_m[3], row3, xptr.add(3 * s_x), b0);
            row4 = cfma_accum(mask_m[4], row4, xptr.add(4 * s_x), b0);
            row5 = cfma_accum(mask_m[5], row5, xptr.add(5 * s_x), b0);
            row6 = cfma_accum(mask_m[6], row6, xptr.add(6 * s_x), b0);
            row7 = cfma_accum(mask_m[7], row7, xptr.add(7 * s_x), b0);
            xptr = xptr.add(1);
        }
        let mut mask_t = mask_m;
        for k in 0..p - d {
            mask_t[k] = 0;
            // //println!("inside the boundary");
            let b0 = mask_load(mask_n, yptr);
            // //println!("b0 {b0:?}");
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
        // //println!("row0 {row0:?}");
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
        let mut row0 = mask_load(mask_n, tptr);
        let mut row1 = mask_load(mask_n, tptr.add(s_t));
        let mut row2 = mask_load(mask_n, tptr.add(s_t * 2));
        let mut row3 = mask_load(mask_n, tptr.add(s_t * 3));
        let mut row4 = mask_load(mask_n, tptr.add(s_t * 4));
        let mut row5 = mask_load(mask_n, tptr.add(s_t * 5));
        let mut row6 = mask_load(mask_n, tptr.add(s_t * 6));
        let mut row7 = mask_load(mask_n, tptr.add(s_t * 7));
        let mask_m = MASK[m];
        let mut mask_t = MASK[0];
        let threshold = m.min(p);
        for k in 0..threshold {
            mask_t[k] = mask_m[k];
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
        for _k in threshold..p {
            // _mm_prefetch(yptr.add(s_y) as *const i8, _MM_HINT_T0);
            // _mm_prefetch(xptr.add(4 * s_x) as *const i8, _MM_HINT_T0);
            let b0 = mask_load(mask_n, yptr);
            yptr = yptr.add(s_y);
            row0 = cfma_accum(mask_m[0], row0, xptr, b0);
            row1 = cfma_accum(mask_m[1], row1, xptr.add(s_x), b0);
            row2 = cfma_accum(mask_m[2], row2, xptr.add(2 * s_x), b0);
            row3 = cfma_accum(mask_m[3], row3, xptr.add(3 * s_x), b0);
            row4 = cfma_accum(mask_m[4], row4, xptr.add(4 * s_x), b0);
            row5 = cfma_accum(mask_m[5], row5, xptr.add(5 * s_x), b0);
            row6 = cfma_accum(mask_m[6], row6, xptr.add(6 * s_x), b0);
            row7 = cfma_accum(mask_m[7], row7, xptr.add(7 * s_x), b0);
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
