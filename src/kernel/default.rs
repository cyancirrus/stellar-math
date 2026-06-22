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
pub fn kernel_tmult_simd(
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
    panic!("not yet implemented");
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
pub fn kernel_lt_mult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    _: usize,
    _: usize,
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
        for i in 0..m {
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
                for j in 0..=i {
                    acc[j] += scalar * *yptr.add(j);
                }
            }
            for j in 0..=i {
                *tptr.add(j) += acc[j];
            }
            tptr = tptr.add(s_t);
            xptr = xptr.add(s_x);
        }
    }
}
pub fn kernel_ut_mult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    _: usize,
    _: usize,
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
        for i in 0..m {
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
                for j in 0..=i {
                    acc[j] += scalar * *yptr.add(j);
                }
            }
            for j in 0..=i {
                *tptr.add(j) += acc[j];
            }
            tptr = tptr.add(s_t);
            xptr = xptr.add(s_x);
        }
    }
}
pub fn kernel_tlt_mult_simd(
    _xptr: *const f32,
    _yptr: *const f32,
    _tptr: *mut f32,
    _: usize,
    _: usize,
    _m: usize,
    _p: usize,
    _n: usize,
    _s_x: usize,
    _s_y: usize,
    _s_t: usize,
) {
    panic!("not yet implemented");
}
pub fn kernel_rlt_mult_simd(
    _xptr: *const f32,
    _yptr: *const f32,
    _tptr: *mut f32,
    _: usize,
    _: usize,
    _m: usize,
    _p: usize,
    _n: usize,
    _s_x: usize,
    _s_y: usize,
    _s_t: usize,
) {
    panic!("not yet implemented");
}
pub fn kernel_rut_mult_simd(
    _xptr: *const f32,
    _yptr: *const f32,
    _tptr: *mut f32,
    _: usize,
    _: usize,
    _m: usize,
    _p: usize,
    _n: usize,
    _s_x: usize,
    _s_y: usize,
    _s_t: usize,
) {
    panic!("not yet implemented");
}
pub fn kernel_tut_mult_simd(
    _xptr: *const f32,
    _yptr: *const f32,
    _tptr: *mut f32,
    _: usize,
    _: usize,
    _m: usize,
    _p: usize,
    _n: usize,
    _s_x: usize,
    _s_y: usize,
    _s_t: usize,
) {
    panic!("not yet implemented");
}
