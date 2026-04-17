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
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    block_m: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // NOTE: might be able to dual accumulate so that the acc isn't blocked
    // default rust method
    unsafe {
        let yorig = yptr;
        let mut acc = [0f32; 8];
        for _i in 0..block_m {
            yptr = yorig;
            let scalar = *xptr;
            for j in 0..SIMD_WIDTH {
                acc[j] = scalar * *yptr.add(j);
            }
            for k in 1..SIMD_WIDTH {
                yptr = yptr.add(s_y);
                let scalar = *xptr.add(k);
                for j in 0..SIMD_WIDTH {
                    acc[j] += scalar * *yptr.add(j);
                }
            }
            for j in 0..SIMD_WIDTH {
                *tptr.add(j) += acc[j];
            }
            xptr = xptr.add(s_x);
            tptr = tptr.add(s_t);
        }
    }
}
#[inline(always)]
pub fn kernel_mult_scalar(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    block_m: usize,
    block_p: usize,
    block_n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        // simple method to handle edge cases
        let yorig = yptr;
        let mut acc = [0f32; SIMD_WIDTH];
        for _i in 0..block_m {
            yptr = yorig;
            let scalar = *xptr;
            for j in 0..block_n {
                acc[j] = scalar * *yptr.add(j);
            }
            for k in 1..block_p {
                yptr = yptr.add(s_y);
                let scalar = *xptr.add(k);
                for j in 0..block_n {
                    acc[j] += scalar * *yptr.add(j);
                }
            }
            for j in 0..block_n {
                *tptr.add(j) += acc[j];
            }
            xptr = xptr.add(s_x);
            tptr = tptr.add(s_t);
        }
    }
}
