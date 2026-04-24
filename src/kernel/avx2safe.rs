// #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use std::arch::x86_64::{
    __m256, __m256i, _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_si256,
    _mm256_maskload_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_loadu_ps
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
unsafe fn gate_row(ptr: *const f32, cur: usize, cap: usize, mask: __m256i) -> __m256 {
    unsafe {
        if cur < cap {
            _mm256_maskload_ps(ptr, mask)
        } else {
            _mm256_setzero_ps()
            // _mm256_loadu_ps(&0f32 as *const f32)
        }
    }
}

#[target_feature(enable = "avx,fma")]
pub fn kernel_mult_safe(
    mut xptr: *const f32,
    yptr: *const f32,
    tptr: *mut f32,
    mut wptr: *mut f32,
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
        let wbase = wptr;
        let mask_n_ptr = MASK[n].as_ptr() as *const __m256i;
        let mask_n = _mm256_loadu_si256(mask_n_ptr);
        let i_row = gate_row(yptr, 0, p, mask_n);
        let ii_row = gate_row(yptr.add(s_y), 1, p, mask_n);
        let iii_row = gate_row(yptr.add(s_y * 2), 2, p, mask_n);
        let iv_row = gate_row(yptr.add(s_y * 3), 3, p, mask_n);
        let v_row = gate_row(yptr.add(s_y * 4), 4, p, mask_n);
        let vi_row = gate_row(yptr.add(s_y * 5), 5, p, mask_n);
        let vii_row = gate_row(yptr.add(s_y * 6), 6, p, mask_n);
        let viii_row = gate_row(yptr.add(s_y * 7), 7, p, mask_n);

        for _ in 0..m {
            let mut acc1 = _mm256_setzero_ps();
            let mut acc0 = _mm256_setzero_ps();
            _mm_prefetch(xptr.add(s_x) as *const i8, _MM_HINT_T0);
            _mm_prefetch(wptr.add(8) as *const i8, _MM_HINT_T0);
            // start with existing t for accumulation
            acc0 = _mm256_fmadd_ps(gate_value(xptr, 0, p), i_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr.add(1), 1, p), ii_row, acc1);
            acc0 = _mm256_fmadd_ps(gate_value(xptr.add(2), 2, p), iii_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr.add(3), 3, p), iv_row, acc1);
            acc0 = _mm256_fmadd_ps(gate_value(xptr.add(4), 4, p), v_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr.add(5), 5, p), vi_row, acc1);
            acc0 = _mm256_fmadd_ps(gate_value(xptr.add(6), 6, p), vii_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr.add(7), 7, p), viii_row, acc1);
            _mm256_storeu_ps(wptr, _mm256_add_ps(acc1, acc0));
            xptr = xptr.add(s_x);
            wptr = wptr.add(8);
        }
        wptr = wbase;
        let (mut tidx, mut widx) = (0, 0);
        for _ in 0..m {
            for k in 0..n {
                *tptr.add(tidx + k) += *wptr.add(widx + k);
            }
            widx += 8;
            tidx += s_t;
        }
    }
}

#[target_feature(enable = "avx,fma")]
pub fn kernel_imult_safe(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    wptr: *mut f32,
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
        let mut i_row = gate_row(wptr, 0, p, mask_n);
        let mut ii_row = gate_row(wptr.add(s_t), 1, p, mask_n);
        let mut iii_row = gate_row(wptr.add(s_t * 2), 2, p, mask_n);
        let mut iv_row = gate_row(wptr.add(s_t * 3), 3, p, mask_n);
        let mut v_row = gate_row(wptr.add(s_t * 4), 4, p, mask_n);
        let mut vi_row = gate_row(wptr.add(s_t * 5), 5, p, mask_n);
        let mut vii_row = gate_row(wptr.add(s_t * 6), 6, p, mask_n);
        let mut viii_row = gate_row(wptr.add(s_t * 7), 7, p, mask_n);
        for _ in 0..p {
            _mm_prefetch(yptr.add(s_y) as *const i8, _MM_HINT_T0);
            let b = _mm256_maskload_ps(yptr, mask_n);
            i_row = _mm256_fmadd_ps(gate_value(xptr, 0, m), b, i_row);
            ii_row = _mm256_fmadd_ps(gate_value(xptr.add(s_x), 1, m), b, ii_row);
            iii_row = _mm256_fmadd_ps(gate_value(xptr.add(2 * s_x), 2, m), b, iii_row);
            iv_row = _mm256_fmadd_ps(gate_value(xptr.add(3 * s_x), 3, m), b, iv_row);
            v_row = _mm256_fmadd_ps(gate_value(xptr.add(4 * s_x), 4, m), b, v_row);
            vi_row = _mm256_fmadd_ps(gate_value(xptr.add(5 * s_x), 5, m), b, vi_row);
            vii_row = _mm256_fmadd_ps(gate_value(xptr.add(6 * s_x), 6, m), b, vii_row);
            viii_row = _mm256_fmadd_ps(gate_value(xptr.add(7 * s_x), 7, m), b, viii_row);
            // accumulates k offset
            xptr = xptr.add(s_x + 1);
            yptr = yptr.add(s_y);
        }
        let (mut tidx, mut widx) = (0, 0);
        for _ in 0..m {
            for k in 0..n {
                *tptr.add(tidx + k) += *wptr.add(widx + k);
            }
            widx += 8;
            tidx += s_t;
        }
    }
}
