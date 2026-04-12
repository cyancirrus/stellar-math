const BLOCK_AVX: usize = 8;
use std::arch::x86_64::{
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};
#[target_feature(enable = "avx,fma")]
pub fn kernel_ut_mult_avx(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_v: usize,
    s_x: usize,
    s_y: usize,
) {
    debug_assert!(block_v <= BLOCK_AVX);
    unsafe {
        let aptr = a.as_ptr();
        let bptr = b.as_ptr();
        let cptr = c.as_mut_ptr();
        // let i_row = _mm256_loadu_ps(bptr);
        let irows = [
            _mm256_loadu_ps(bptr),
            _mm256_loadu_ps(bptr.add(s_y)),
            _mm256_loadu_ps(bptr.add(s_y * 2)),
            _mm256_loadu_ps(bptr.add(s_y * 3)),
            _mm256_loadu_ps(bptr.add(s_y * 4)),
            _mm256_loadu_ps(bptr.add(s_y * 5)),
            _mm256_loadu_ps(bptr.add(s_y * 6)),
            _mm256_loadu_ps(bptr.add(s_y * 7)),
        ];
        let mut aoffset = 0;
        // let mut coffset = offset;
        let mut coffset = 0;
        for i in 0..block_v {
            let arow = aptr.add(aoffset);
            let c_row = cptr.add(coffset);
            let mut acc = _mm256_loadu_ps(c_row);
            for k in i..BLOCK_AVX {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(k)), irows[k], acc);
            }
            _mm256_storeu_ps(c_row, acc);
            aoffset += s_x;
            coffset += s_y;
        }
    }
}

#[target_feature(enable = "avx,fma")]
pub fn kernel_lt_mult_avx(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_v: usize,
    s_x: usize,
    s_y: usize,
) {
    debug_assert!(block_v <= BLOCK_AVX);
    unsafe {
        let aptr = a.as_ptr();
        let bptr = b.as_ptr();
        let cptr = c.as_mut_ptr();
        // let i_row = _mm256_loadu_ps(bptr);
        let irows = [
            _mm256_loadu_ps(bptr),
            _mm256_loadu_ps(bptr.add(s_y)),
            _mm256_loadu_ps(bptr.add(s_y * 2)),
            _mm256_loadu_ps(bptr.add(s_y * 3)),
            _mm256_loadu_ps(bptr.add(s_y * 4)),
            _mm256_loadu_ps(bptr.add(s_y * 5)),
            _mm256_loadu_ps(bptr.add(s_y * 6)),
            _mm256_loadu_ps(bptr.add(s_y * 7)),
        ];
        let mut aoffset = 0;
        // let mut coffset = offset;
        let mut coffset = 0;
        for i in 0..block_v {
            let arow = aptr.add(aoffset);
            let c_row = cptr.add(coffset);
            let mut acc = _mm256_loadu_ps(c_row);
            for k in 0..=i {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(k)), irows[k], acc);
            }
            _mm256_storeu_ps(c_row, acc);
            aoffset += s_x;
            coffset += s_y;
        }
    }
}

#[cfg(test)]
mod test_avx2_kernels {
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;

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
            kernel_lt_mult_avx(&a.data, &b.data, output, BLOCK_AVX, m, n);
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
            kernel_ut_mult_avx(&a.data, &b.data, output, BLOCK_AVX, m, n);
        }
        debug_assert!(approx_vector_eq(&expected.data, &output[..m * n]));
    }

    #[test]
    fn test_triangle_equalities() {
        let (m, k, n) = (8, 8, 8);
        let mut output = vec![0f32; m * n];
        test_lower_triangle(m, k, n, &mut output);
        output.fill(0f32);
        test_upper_triangle(m, k, n, &mut output);
        output.fill(0f32);
    }
}
