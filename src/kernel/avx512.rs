#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps,
    _mm512_storeu_ps,
};

#[target_feature(enable = "avx512f,fma")]
pub fn kernel_mult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    block_m: usize,
    s_x: usize,
    s_y: usize,
) {
    unsafe {
        let i_row = _mm512_loadu_ps(yptr);
        let ii_row = _mm512_loadu_ps(yptr.add(s_y));
        let iii_row = _mm512_loadu_ps(yptr.add(s_y * 2));
        let iv_row = _mm512_loadu_ps(yptr.add(s_y * 3));
        let v_row = _mm512_loadu_ps(yptr.add(s_y * 4));
        let vi_row = _mm512_loadu_ps(yptr.add(s_y * 5));
        let vii_row = _mm512_loadu_ps(yptr.add(s_y * 6));
        let viii_row = _mm512_loadu_ps(yptr.add(s_y * 7));
        let ix_row = _mm512_loadu_ps(yptr.add(s_y * 8));
        let x_row = _mm512_loadu_ps(yptr.add(s_y * 9));
        let xi_row = _mm512_loadu_ps(yptr.add(s_y * 10));
        let xii_row = _mm512_loadu_ps(yptr.add(s_y * 11));
        let xiii_row = _mm512_loadu_ps(yptr.add(s_y * 12));
        let xiv_row = _mm512_loadu_ps(yptr.add(s_y * 13));
        let xv_row = _mm512_loadu_ps(yptr.add(s_y * 14));
        let xvi_row = _mm512_loadu_ps(yptr.add(s_y * 15));

        for _ in 0..block_m {
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            // start with existing t for accumulation
            let mut acc3 = _mm512_loadu_ps(tptr);
            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(0)), i_row, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(1)), ii_row, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(2)), iii_row, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(3)), iv_row, acc3);

            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(4)), v_row, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(5)), vi_row, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(6)), vii_row, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(7)), viii_row, acc3);

            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(8)), ix_row, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(9)), x_row, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(10)), xi_row, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(11)), xii_row, acc3);

            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(12)), xiii_row, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(13)), xiv_row, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(14)), xv_row, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(*xptr.add(15)), xvi_row, acc3);

            _mm512_storeu_ps(
                t_row,
                _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)),
            );
            xptr = xptr.add(s_x);
            tptr = tptr.add(s_y);
        }
    }
}
