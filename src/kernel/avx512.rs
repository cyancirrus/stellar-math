use std::arch::x86_64::{
    _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps,
    _mm512_storeu_ps,_mm512_add_ps
};

const BLOCK_AVX512:usize = 16;

#[target_feature(enable = "avx512f,fma")]
pub fn kernel_mult_avx(
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    block_v: usize,
    s_x: usize,
    s_y: usize,
    // offset: usize,
) {
    unsafe {
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();
        let tptr = t.as_mut_ptr();
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

        let mut xoffset = 0;
        let mut toffset = 0;
        for _ in 0..block_v {
            let xrow = xptr.add(xoffset);
            let t_row = tptr.add(toffset);
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_loadu_ps(t_row);
            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(0)), i_row, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(1)), ii_row, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(2)), iii_row, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(3)), iv_row, acc3);

            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(4)), v_row, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(5)), vi_row, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(6)), vii_row, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(7)), viii_row, acc3);

            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(8)), ix_row, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(9)), x_row, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(10)), xi_row, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(11)), xii_row, acc3);

            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(12)), xiii_row, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(13)), xiv_row, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(14)), xv_row, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(*xrow.add(15)), xvi_row, acc3);

            _mm512_storeu_ps(t_row, _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
            // xoffset += BLOCK_AVX512;
            xoffset += s_x;
            toffset += s_y;
        }
    }
}
