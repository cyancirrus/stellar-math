use crate::arch::SIMD_WIDTH;

/// kernel_mult_scalar
/// a * b -> c
///
/// * a : block of a
/// * b : block of b
/// * c : block row of c
/// * block_v : size of block rows which is equal to block cols
/// * stride : the number of cols in the output matrix c
/// * offset : the outer k which will determine where we need to write
#[inline(always)]
pub unsafe fn kernel_mult_simd(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_v: usize,
    s_x: usize,
    s_y: usize,
) {
    // default rust method
    let mut aoffset = 0;
    let mut coffset = 0;
    let mut boffset;
    for _i in 0..block_v {
        boffset = 0;
        let a_row = &a[aoffset..aoffset + s_x];
        for k in 0..SIMD_WIDTH {
            let scalar = a_row[k];
            let b_row = &b[boffset..boffset + SIMD_WIDTH];
            let c_row = &mut c[coffset..coffset + SIMD_WIDTH];
            for (c, b) in c_row.iter_mut().zip(b_row.iter()) {
                *c += scalar * b;
            }
            boffset += s_y;
        }
        aoffset += SIMD_WIDTH;
        coffset += s_y;
    }
}

#[inline(always)]
pub fn kernel_mult_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_m: usize,
    block_k: usize,
    block_n: usize,
    s_x: usize,
    s_y: usize,
) {
    // simple method to handle edge cases
    let mut aoffset = 0;
    let mut coffset = 0;
    let mut boffset;
    for _i in 0..block_m {
        boffset = 0;
        let a_row = &a[aoffset..aoffset + s_x];
        for k in 0..block_k {
            let scalar = a_row[k];
            let b_row = &b[boffset..boffset + block_n];
            let c_row = &mut c[coffset..coffset + block_n];
            for (c, b) in c_row.iter_mut().zip(b_row.iter()) {
                *c += scalar * b;
            }
            boffset += s_y;
        }
        aoffset += SIMD_WIDTH;
        coffset += s_y;
    }
}
