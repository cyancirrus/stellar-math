use crate::arch::SIMD_WIDTH;

// NOTE: could consider switching to a mask look up table

/// kernel_mult_scalar
/// a * b -> c
///
/// * a : block of a
/// * b : block of b
/// * c : block row of c
/// * v : size of block rows which is equal to block cols
/// * stride : the number of cols in the output matrix c
/// * offset : the outer k which will determine where we need to write
#[inline(always)]
pub fn kernel_mult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut acc = [0f32; 8];
        let yorig = yptr;
        for _ in 0..m {
            yptr = yorig;
            {
                let scalar = *xptr;
                for j in 0..n {
                    acc[j] = scalar * *yptr.add(j);
                }
            }
            for k in 1..p {
                let scalar = *xptr.add(k);
                yptr = yptr.add(s_y);
                for j in 0..n {
                    acc[j] += scalar * *yptr.add(j);
                }
            }
            for j in 0..n {
                *tptr.add(j) += acc[j];
            }
            tptr = tptr.add(s_t);
            xptr = xptr.add(s_x);
        }
    }
}
#[inline(always)]
pub unsafe fn kernel_mult_simd_aligned(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    m: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // NOTE: might be able to dual accumulate so that the acc isn't blocked
    // default rust method
    unsafe {
        let mut acc = [0f32; 8];
        let yorig = yptr;
        for _ in 0..m {
            {
                let scalar = *xptr;
                for j in 0..SIMD_WIDTH {
                    acc[j] = scalar * *yptr.add(j);
                }
            }
            for k in 1..SIMD_WIDTH {
                let scalar = *xptr.add(k);
                yptr = yptr.add(s_y);
                for j in 0..SIMD_WIDTH {
                    acc[j] += scalar * *yptr.add(j);
                }
            }
            for j in 0..SIMD_WIDTH {
                *tptr.add(j) += acc[j];
            }
            tptr = tptr.add(s_t);
            xptr = xptr.add(s_x);
            yptr = yorig;
        }
    }
}
#[inline(always)]
pub fn kernel_mult_scalar(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut acc = [0f32; 8];
        let yorig = yptr;
        for _ in 0..m {
            yptr = yorig;
            {
                let scalar = *xptr;
                for j in 0..n {
                    acc[j] = scalar * *yptr.add(j);
                }
            }
            for k in 1..p {
                let scalar = *xptr.add(k);
                yptr = yptr.add(s_y);
                for j in 0..n {
                    acc[j] += scalar * *yptr.add(j);
                }
            }
            for j in 0..n {
                *tptr.add(j) += acc[j];
            }
            tptr = tptr.add(s_t);
            xptr = xptr.add(s_x);
        }
    }
}
