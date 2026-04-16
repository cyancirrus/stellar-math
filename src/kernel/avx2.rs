#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};
/// kernel_mult
/// a * b -> c
///
/// * a : block of a
/// * b : block of b
/// * c : block row of c
/// * block_m : size of block rows which is equal to block cols
/// * s_x : number of columns in input matrix x
/// * s_y : number of columns in input matrix y
#[target_feature(enable = "avx,fma")]
pub fn kernel_mult_simd(
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    block_m: usize,
    s_x: usize,
    s_y: usize,
) {
    unsafe {
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();
        let tptr = t.as_mut_ptr();
        let i_row = _mm256_loadu_ps(yptr);
        let ii_row = _mm256_loadu_ps(yptr.add(s_y));
        let iii_row = _mm256_loadu_ps(yptr.add(s_y * 2));
        let iv_row = _mm256_loadu_ps(yptr.add(s_y * 3));
        let v_row = _mm256_loadu_ps(yptr.add(s_y * 4));
        let vi_row = _mm256_loadu_ps(yptr.add(s_y * 5));
        let vii_row = _mm256_loadu_ps(yptr.add(s_y * 6));
        let viii_row = _mm256_loadu_ps(yptr.add(s_y * 7));

        let mut xoffset = 0;
        let mut toffset = 0;
        for _ in 0..block_m {
            let xrow = xptr.add(xoffset);
            let trow = tptr.add(toffset);
            let mut acc0 = _mm256_setzero_ps();
            // start with existing t for accumulation
            let mut acc1 = _mm256_loadu_ps(trow);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(0)), i_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(1)), ii_row, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(2)), iii_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(3)), iv_row, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(4)), v_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(5)), vi_row, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(6)), vii_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(7)), viii_row, acc1);
            _mm256_storeu_ps(trow, _mm256_add_ps(acc0, acc1));
            xoffset += s_x;
            toffset += s_y;
        }
    }
}
#[target_feature(enable = "avx,fma")]
pub fn kernel_ut_mult_simd(
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    block_m: usize,
    s_x: usize,
    s_y: usize,
) {
    debug_assert!(block_m <= 8);
    unsafe {
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();
        let tptr = t.as_mut_ptr();
        // let i_row = _mm256_loadu_ps(yptr);
        let irows = [
            _mm256_loadu_ps(yptr),
            _mm256_loadu_ps(yptr.add(s_y)),
            _mm256_loadu_ps(yptr.add(s_y * 2)),
            _mm256_loadu_ps(yptr.add(s_y * 3)),
            _mm256_loadu_ps(yptr.add(s_y * 4)),
            _mm256_loadu_ps(yptr.add(s_y * 5)),
            _mm256_loadu_ps(yptr.add(s_y * 6)),
            _mm256_loadu_ps(yptr.add(s_y * 7)),
        ];
        let mut xoffset = 0;
        let mut toffset = 0;
        for i in 0..block_m {
            let xrow = xptr.add(xoffset);
            let trow = tptr.add(toffset);
            let mut acc = _mm256_loadu_ps(trow);
            for k in i..8 {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(k)), irows[k], acc);
            }
            _mm256_storeu_ps(trow, acc);
            xoffset += s_x;
            toffset += s_y;
        }
    }
}

#[target_feature(enable = "avx,fma")]
pub fn kernel_lt_mult_simd(
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    block_m: usize,
    s_x: usize,
    s_y: usize,
) {
    debug_assert!(block_m <= 8);
    unsafe {
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();
        let tptr = t.as_mut_ptr();
        let irows = [
            _mm256_loadu_ps(yptr),
            _mm256_loadu_ps(yptr.add(s_y)),
            _mm256_loadu_ps(yptr.add(s_y * 2)),
            _mm256_loadu_ps(yptr.add(s_y * 3)),
            _mm256_loadu_ps(yptr.add(s_y * 4)),
            _mm256_loadu_ps(yptr.add(s_y * 5)),
            _mm256_loadu_ps(yptr.add(s_y * 6)),
            _mm256_loadu_ps(yptr.add(s_y * 7)),
        ];
        let mut xoffset = 0;
        let mut toffset = 0;
        for i in 0..block_m {
            let xrow = xptr.add(xoffset);
            let trow = tptr.add(toffset);
            let mut acc = _mm256_loadu_ps(trow);
            for k in 0..=i {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(k)), irows[k], acc);
            }
            _mm256_storeu_ps(trow, acc);
            xoffset += s_x;
            toffset += s_y;
        }
    }
}

#[cfg(test)]
mod test_avx2_kernels {
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;
    const BLOCK_AVX2: usize = 8;

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
            kernel_lt_mult_simd(&a.data, &b.data, output, BLOCK_AVX2, m, n);
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
            kernel_ut_mult_simd(&a.data, &b.data, output, BLOCK_AVX2, m, n);
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
