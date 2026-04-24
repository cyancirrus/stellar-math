// #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use std::arch::x86_64::{
    __m256, __m256i, _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_si256,
    _mm256_maskload_ps, _mm256_maskstore_ps, _mm256_set1_ps, _mm256_setzero_ps,
};

// negative 1 is twos complement so all bits active
#[rustfmt::skip]
const MASK:[[i32;8];9] = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0,  0,  0],
    [-1, -1,  0,  0,  0,  0,  0,  0],
    [-1, -1, -1,  0,  0,  0,  0,  0],
    [-1, -1, -1, -1,  0,  0,  0,  0],
    [-1, -1, -1, -1, -1,  0,  0,  0],
    [-1, -1, -1, -1, -1, -1,  0,  0],
    [-1, -1, -1, -1, -1, -1, -1,  0],
    [-1, -1, -1, -1, -1, -1, -1, -1],
];

unsafe fn gate_value(ptr: *const f32, cur: usize, cap: usize) -> __m256 {
    unsafe {
        let val = if cur < cap { *ptr } else { 0f32 };
        _mm256_set1_ps(val)
    }
}
// unsafe fn gate_row(ptr: *const f32, cur: usize, cap: usize, mask: __m256i) -> __m256 {
//     unsafe {
//         if cur < cap {
//             _mm256_maskload_ps(ptr, mask)
//         } else {
//             _mm256_setzero_ps()
//         }
//     }
// }
unsafe fn sgate_row(ptr: *mut f32, cur: usize, cap: usize, mask: __m256i, data: __m256) {
    unsafe {
        if cur < cap {
            _mm256_maskstore_ps(ptr, mask, data);
        }
    }
}

#[target_feature(enable = "avx,fma")]
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
        let mask_n_ptr = MASK[n].as_ptr() as *const __m256i;
        let mask_n = _mm256_loadu_si256(mask_n_ptr);
        let i_row = _mm256_maskload_ps(yptr, mask_n);
        let ii_row = _mm256_maskload_ps(yptr.add(s_y), mask_n);
        let iii_row = _mm256_maskload_ps(yptr.add(s_y * 2), mask_n);
        let iv_row = _mm256_maskload_ps(yptr.add(s_y * 3), mask_n);
        let v_row = _mm256_maskload_ps(yptr.add(s_y * 4), mask_n);
        let vi_row = _mm256_maskload_ps(yptr.add(s_y * 5), mask_n);
        let vii_row = _mm256_maskload_ps(yptr.add(s_y * 6), mask_n);
        let viii_row = _mm256_maskload_ps(yptr.add(s_y * 7), mask_n);
        for _ in 0..m {
            let mut acc1 = _mm256_maskload_ps(tptr, mask_n);
            let mut acc0 = _mm256_setzero_ps();
            _mm_prefetch(xptr.add(s_x) as *const i8, _MM_HINT_T0);
            _mm_prefetch(tptr.add(s_t) as *const i8, _MM_HINT_T0);
            // start with existing t for accumulation
            acc0 = _mm256_fmadd_ps(gate_value(xptr, 0, p), i_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr.add(1), 1, p), ii_row, acc1);
            acc0 = _mm256_fmadd_ps(gate_value(xptr.add(2), 2, p), iii_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr.add(3), 3, p), iv_row, acc1);
            acc0 = _mm256_fmadd_ps(gate_value(xptr.add(4), 4, p), v_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr.add(5), 5, p), vi_row, acc1);
            acc0 = _mm256_fmadd_ps(gate_value(xptr.add(6), 6, p), vii_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr.add(7), 7, p), viii_row, acc1);
            _mm256_maskstore_ps(tptr, mask_n, _mm256_add_ps(acc1, acc0));
            xptr = xptr.add(s_x);
            tptr = tptr.add(s_t);
        }
    }
}
#[target_feature(enable = "avx,fma")]
pub fn kernel_imult_safe(
    xptr: *const f32,
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
        let mut i_row = _mm256_maskload_ps(tptr, mask_n);
        let mut ii_row = _mm256_maskload_ps(tptr.add(s_t), mask_n);
        let mut iii_row = _mm256_maskload_ps(tptr.add(s_t * 2), mask_n);
        let mut iv_row = _mm256_maskload_ps(tptr.add(s_t * 3), mask_n);
        let mut v_row = _mm256_maskload_ps(tptr.add(s_t * 4), mask_n);
        let mut vi_row = _mm256_maskload_ps(tptr.add(s_t * 5), mask_n);
        let mut vii_row = _mm256_maskload_ps(tptr.add(s_t * 6), mask_n);
        let mut viii_row = _mm256_maskload_ps(tptr.add(s_t * 7), mask_n);
        for k in 0..p {
            _mm_prefetch(yptr.add(s_y) as *const i8, _MM_HINT_T0);
            let b = _mm256_maskload_ps(yptr, mask_n);
            i_row = _mm256_fmadd_ps(gate_value(xptr.add(k), 0, m), b, i_row);
            ii_row = _mm256_fmadd_ps(gate_value(xptr.add(s_x + k), 1, m), b, ii_row);
            iii_row = _mm256_fmadd_ps(gate_value(xptr.add(2 * s_x + k), 2, m), b, iii_row);
            iv_row = _mm256_fmadd_ps(gate_value(xptr.add(3 * s_x + k), 3, m), b, iv_row);
            v_row = _mm256_fmadd_ps(gate_value(xptr.add(4 * s_x + k), 4, m), b, v_row);
            vi_row = _mm256_fmadd_ps(gate_value(xptr.add(5 * s_x + k), 5, m), b, vi_row);
            vii_row = _mm256_fmadd_ps(gate_value(xptr.add(6 * s_x + k), 6, m), b, vii_row);
            viii_row = _mm256_fmadd_ps(gate_value(xptr.add(7 * s_x + k), 7, m), b, viii_row);
            // accumulates k offset
            yptr = yptr.add(s_y);
        }
        sgate_row(tptr, 0, m, mask_n, i_row);
        sgate_row(tptr.add(s_t), 1, m, mask_n, ii_row);
        sgate_row(tptr.add(s_t * 2), 2, m, mask_n, iii_row);
        sgate_row(tptr.add(s_t * 3), 3, m, mask_n, iv_row);
        sgate_row(tptr.add(s_t * 4), 4, m, mask_n, v_row);
        sgate_row(tptr.add(s_t * 5), 5, m, mask_n, vi_row);
        sgate_row(tptr.add(s_t * 6), 6, m, mask_n, vii_row);
        sgate_row(tptr.add(s_t * 7), 7, m, mask_n, viii_row);
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
            assert!(approx_vector_eq(&expect.data, &t));
            let mut x_simd = x.data.clone();
            let mut y_simd = y.data.clone();
            let mut w = vec![0f32; 8 * 8];
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
