#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use crate::arch::SIMD_WIDTH;
use crate::kernel::avx2safe;
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
    unsafe {
        if (m | n) & (SIMD_WIDTH - 1) == 0 {
            kernel_imult_simd_aligned(xptr, yptr, tptr, p, s_x, s_y, s_t);
        // } else if (p | n) & (SIMD_WIDTH - 1) == 0 {
        //     kernel_mult_simd_aligned(xptr, yptr, tptr, m, s_x, s_y, s_t);
        } else {
            // avx2safe::kernel_mult_safe(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
            avx2safe::kernel_imult_safe(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
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
        let i_row = _mm256_loadu_ps(yptr);
        let v_row = _mm256_loadu_ps(yptr.add(s_y * 4));
        let ii_row = _mm256_loadu_ps(yptr.add(s_y));
        let vi_row = _mm256_loadu_ps(yptr.add(s_y * 5));
        let iii_row = _mm256_loadu_ps(yptr.add(s_y * 2));
        let vii_row = _mm256_loadu_ps(yptr.add(s_y * 6));
        let iv_row = _mm256_loadu_ps(yptr.add(s_y * 3));
        let viii_row = _mm256_loadu_ps(yptr.add(s_y * 7));
        // t is being passed in
        for _ in 0..m {
            let mut acc1 = _mm256_loadu_ps(tptr);
            let mut acc0 = _mm256_setzero_ps();
            // _mm_prefetch(xptr.add(s_x) as *const i8, _MM_HINT_T0);
            // _mm_prefetch(tptr.add(s_t) as *const i8, _MM_HINT_T0);
            // start with existing t for accumulation
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr), i_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(4)), v_row, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(1)), ii_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(5)), vi_row, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(2)), iii_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(6)), vii_row, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(3)), iv_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(7)), viii_row, acc1);
            _mm256_storeu_ps(tptr, _mm256_add_ps(acc1, acc0));
            xptr = xptr.add(s_x);
            tptr = tptr.add(s_t);
        }
    }
}
#[target_feature(enable = "avx,avx2,fma")]
pub fn kernel_imult_simd_aligned(
    xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    p: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // Sum[K] Union[I] { g^i = aik b^k }
    // excels at processing panels of data ie 8 x K * K x 8;
    unsafe {
        let mut i_row = _mm256_loadu_ps(tptr);
        let mut v_row = _mm256_loadu_ps(tptr.add(s_t * 4));
        let mut ii_row = _mm256_loadu_ps(tptr.add(s_t));
        let mut vi_row = _mm256_loadu_ps(tptr.add(s_t * 5));
        let mut iii_row = _mm256_loadu_ps(tptr.add(s_t * 2));
        let mut vii_row = _mm256_loadu_ps(tptr.add(s_t * 6));
        let mut iv_row = _mm256_loadu_ps(tptr.add(s_t * 3));
        let mut viii_row = _mm256_loadu_ps(tptr.add(s_t * 7));
        for k in 0..p {
            let b = _mm256_loadu_ps(yptr);
            i_row = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(k)), b, i_row);
            v_row = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(4 * s_x + k)), b, v_row);
            ii_row = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(s_x + k)), b, ii_row);
            vi_row = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(5 * s_x + k)), b, vi_row);
            iii_row = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(2 * s_x + k)), b, iii_row);
            vii_row = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(6 * s_x + k)), b, vii_row);
            iv_row = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(3 * s_x + k)), b, iv_row);
            viii_row = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xptr.add(7 * s_x + k)), b, viii_row);
            yptr = yptr.add(s_y);
        }
        _mm256_storeu_ps(tptr, i_row);
        _mm256_storeu_ps(tptr.add(s_t * 4), v_row);
        _mm256_storeu_ps(tptr.add(s_t), ii_row);
        _mm256_storeu_ps(tptr.add(s_t * 5), vi_row);
        _mm256_storeu_ps(tptr.add(s_t * 2), iii_row);
        _mm256_storeu_ps(tptr.add(s_t * 6), vii_row);
        _mm256_storeu_ps(tptr.add(s_t * 3), iv_row);
        _mm256_storeu_ps(tptr.add(s_t * 7), viii_row);
    }
}
#[rustfmt::skip]
#[target_feature(enable = "avx,fma")]
// pub fn kernel_trans_simd(mut tptr: *mut f32, mut wptr: *mut f32) {
pub fn kernel_trans_simd(mut tptr: *mut f32) {
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
        // _mm256_storeu_ps(wptr, _mm256_permute2f128_ps(q0, q4, 0x20));
        // _mm256_storeu_ps(wptr.add(8), _mm256_permute2f128_ps(q0, q4, 0x31));
        // _mm256_storeu_ps(wptr.add(8 * 2), _mm256_permute2f128_ps(q1, q5, 0x20));
        // _mm256_storeu_ps(wptr.add(8 * 3), _mm256_permute2f128_ps(q1, q5, 0x31));
        // _mm256_storeu_ps(wptr.add(8 * 4), _mm256_permute2f128_ps(q2, q6, 0x20));
        // _mm256_storeu_ps(wptr.add(8 * 5), _mm256_permute2f128_ps(q2, q6, 0x31));
        // _mm256_storeu_ps(wptr.add(8 * 6), _mm256_permute2f128_ps(q3, q7, 0x20));
        // _mm256_storeu_ps(wptr.add(8 * 7), _mm256_permute2f128_ps(q3, q7, 0x31));
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
#[target_feature(enable = "avx,fma")]
pub fn kernel_wmult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // Sum[K] Union[I] { g^i = aik b^k }
    unsafe {
        let mut i_row_0 = _mm256_loadu_ps(tptr);
        let mut i_row_1 = _mm256_loadu_ps(tptr.add(s_t));
        let mut ii_row_0 = _mm256_loadu_ps(tptr.add(s_t * 2));
        let mut ii_row_1 = _mm256_loadu_ps(tptr.add(s_t * 3));
        let mut iii_row_0 = _mm256_loadu_ps(tptr.add(s_t * 4));
        let mut iii_row_1 = _mm256_loadu_ps(tptr.add(s_t * 5));
        let mut iv_row_0 = _mm256_loadu_ps(tptr.add(s_t * 6));
        let mut iv_row_1 = _mm256_loadu_ps(tptr.add(s_t * 7));
        // t is being passed in
        for _ in 0..4 {
            _mm_prefetch(yptr.add(s_y) as *const i8, _MM_HINT_T0);
            let b_0 = _mm256_loadu_ps(yptr);
            let b_1 = _mm256_loadu_ps(yptr.add(8));
            i_row_0 = _mm256_fmadd_ps(_mm256_set1_ps(*xptr), b_0, i_row_0);
            i_row_1 = _mm256_fmadd_ps(_mm256_set1_ps(*xptr), b_1, i_row_1);
            ii_row_0 = _mm256_fmadd_ps(_mm256_set1_ps(*xptr.add(s_x)), b_0, ii_row_0);
            ii_row_1 = _mm256_fmadd_ps(_mm256_set1_ps(*xptr.add(s_x)), b_1, ii_row_1);
            iii_row_0 = _mm256_fmadd_ps(_mm256_set1_ps(*xptr.add(2 * s_x)), b_0, iii_row_0);
            iii_row_1 = _mm256_fmadd_ps(_mm256_set1_ps(*xptr.add(2 * s_x)), b_1, iii_row_1);
            iv_row_0 = _mm256_fmadd_ps(_mm256_set1_ps(*xptr.add(3 * s_x)), b_0, iv_row_0);
            iv_row_1 = _mm256_fmadd_ps(_mm256_set1_ps(*xptr.add(3 * s_x)), b_1, iv_row_1);
            // accumulates k offset
            xptr = xptr.add(s_x + 1);
            yptr = yptr.add(s_y);
        }
        _mm256_storeu_ps(tptr, i_row_0);
        _mm256_storeu_ps(tptr.add(8), i_row_1);
        _mm256_storeu_ps(tptr.add(s_t), ii_row_0);
        _mm256_storeu_ps(tptr.add(s_t + 8), ii_row_1);
        _mm256_storeu_ps(tptr.add(2 * s_t), iii_row_0);
        _mm256_storeu_ps(tptr.add(2 * s_t + 8), iii_row_1);
        _mm256_storeu_ps(tptr.add(3 * s_t), iv_row_0);
        _mm256_storeu_ps(tptr.add(3 * s_t + 8), iv_row_1);
    }
}
#[target_feature(enable = "avx,fma")]
pub fn kernel_ut_mult_simd(
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    m: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    debug_assert!(m <= 8);
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
        for i in 0..m {
            let xrow = xptr.add(xoffset);
            let trow = tptr.add(toffset);
            let mut acc = _mm256_loadu_ps(trow);
            for k in i..8 {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(k)), irows[k], acc);
            }
            _mm256_storeu_ps(trow, acc);
            xoffset += s_x;
            toffset += s_t;
        }
    }
}

#[target_feature(enable = "avx,fma")]
pub fn kernel_lt_mult_simd(
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    m: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    debug_assert!(m <= 8);
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
        for i in 0..m {
            let xrow = xptr.add(xoffset);
            let trow = tptr.add(toffset);
            let mut acc = _mm256_loadu_ps(trow);
            for k in 0..=i {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(*xrow.add(k)), irows[k], acc);
            }
            _mm256_storeu_ps(trow, acc);
            xoffset += s_x;
            toffset += s_t;
        }
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
            kernel_lt_mult_simd(&a.data, &b.data, output, BLOCK_AVX2, m, n, n);
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
            kernel_ut_mult_simd(&a.data, &b.data, output, BLOCK_AVX2, m, n, n);
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
