use crate::arch::SIMD_WIDTH;
use crate::kernel::avx2::{alligned, ltriangle, unalligned};
// #[inline(always)]
pub fn kernel_mult_simd(
    xptr: *const f32,
    yptr: *const f32,
    tptr: *mut f32,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // happens over k-contraction needs the imult kernel
    unsafe {
        if (m | n) & (SIMD_WIDTH - 1) == 0 {
            alligned::kernel_imult_simd_aligned(xptr, yptr, tptr, p, s_x, s_y, s_t);
        } else {
            unalligned::kernel_imult_safe(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
pub fn kernel_lt_mult_simd(
    xptr: *const f32,
    yptr: *const f32,
    tptr: *mut f32,
    d: isize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    let pre = (-d).max(0) as usize;
    let pro = d.max(0) as usize;
    let pos = if d <= 0 {
        m - pre
    } else if d < p as isize {
        (p.wrapping_sub(pro)).min(SIMD_WIDTH)
    } else {
        0
    };
    unsafe {
        if pos > 0 {
            ltriangle::lmult_lt(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        } else {
            kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
pub fn kernel_ut_mult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    d_add: usize,
    d_sub: usize,
    m: usize,
    mut p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    let (d_pos, d_neg) = if d_add > d_sub {
        (d_add - d_sub, 0 )
    } else {
        (0, d_sub - d_add)
    };
    // pre-allign left boundary point
    let pre = m.min(d_neg);
    // handle triangle part of upper triangular
    let pos = (p - p.min(d_pos)).min(m - pre);
    // process the dense part
    let pro = p - d_pos - pos;
    println!("---------------------");
    println!("m {m:}, p: {p:}, n: {n:}"); 
    println!(" - - - - - - - - - - ");
    println!("d_pos {d_pos:}");
    println!("d_neg {d_neg:}");
    println!(" - - - - - - - - - - ");
    println!("pre {pre:}");
    println!("pos {pos:}");
    println!("pro {pro:}");
    println!("---------------------");
    unsafe {
        p = p - d_pos;
        xptr = xptr.add(d_pos);
        yptr = yptr.add(d_pos * s_y);
        // tptr = tptr.add(d_pos);
        if pos > 0 {
            ltriangle::lmult_ut(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        } else {
            println!("dense");
            kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
