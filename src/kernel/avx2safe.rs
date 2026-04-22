use std::arch::x86_64::{
    __m256, __m256i, _MM_HINT_T0, _mm_prefetch, _mm256_add_ps,
    _mm256_fmadd_ps, _mm256_loadu_si256,
    _mm256_maskload_ps,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
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
        let val = if cur < cap { *ptr.add(cur) } else { 0f32 };
        _mm256_set1_ps(val)
    }
}
unsafe fn gate_row( ptr: *const f32, cur: usize, cap: usize, mask: __m256i) -> __m256 {
    unsafe{
        let val = if cur < cap { ptr.add(cur) } else { &0f32  as *const f32};
        _mm256_maskload_ps(val, mask)
    }
}

#[target_feature(enable = "avx,fma")]
pub fn kernel_mult_safe(
    mut xptr: *const f32,
    yptr: *const f32,
    tptr: *mut f32,
    mut wptr: *mut f32,
    block_m: usize,
    block_p: usize,
    block_n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // w: workspace
    // excels at tall x matrix and wide y
    unsafe {
        let wbase = wptr;
        let mask_n_ptr = MASK[block_n].as_ptr() as *const __m256i;
        let mask_n = _mm256_loadu_si256(mask_n_ptr);
        let i_row = gate_row(yptr, 0, block_p, mask_n);
        let ii_row = gate_row(yptr.add(s_y), 1, block_p, mask_n);
        let iii_row = gate_row(yptr.add(s_y * 2), 2, block_p, mask_n);
        let iv_row = gate_row(yptr.add(s_y * 3), 3, block_p, mask_n);
        let v_row = gate_row(yptr.add(s_y * 4), 4, block_p, mask_n);
        let vi_row = gate_row(yptr.add(s_y * 5), 5, block_p, mask_n);
        let vii_row = gate_row(yptr.add(s_y * 6), 6, block_p, mask_n);
        let viii_row = gate_row(yptr.add(s_y * 7), 7, block_p, mask_n);

        for _ in 0..block_m {
            let mut acc1 = _mm256_setzero_ps();
            let mut acc0 = _mm256_setzero_ps();
            _mm_prefetch(xptr.add(s_x) as *const i8, _MM_HINT_T0);
            _mm_prefetch(wptr.add(8) as *const i8, _MM_HINT_T0);
            // start with existing t for accumulation
            acc0 = _mm256_fmadd_ps(gate_value(xptr, 0, block_p), i_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr, 1, block_p), ii_row, acc1);
            acc0 = _mm256_fmadd_ps(gate_value(xptr, 2, block_p), iii_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr, 3, block_p), iv_row, acc1);
            acc0 = _mm256_fmadd_ps(gate_value(xptr, 4, block_p), v_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr, 5, block_p), vi_row, acc1);
            acc0 = _mm256_fmadd_ps(gate_value(xptr, 6, block_p), vii_row, acc0);
            acc1 = _mm256_fmadd_ps(gate_value(xptr, 7, block_p), viii_row, acc1);
            _mm256_storeu_ps(wptr, _mm256_add_ps(acc1, acc0));
            xptr = xptr.add(s_x);
            wptr = wptr.add(8);
        }
        wptr = wbase;
        let (mut tidx, mut widx) = (0, 0);
        for _ in 0..block_m {
            for k in 0..block_n {
                *tptr.add(tidx + k) += *wptr.add(widx + k);
            }
            widx += 8;
            tidx += s_t;
        }
    }
}
