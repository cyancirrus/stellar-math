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
    xptr: *const f32,
    yptr: *const f32,
    tptr: *mut f32,
    d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
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
    let pre = d_neg.min(m);
    // handle triangle part of upper triangular
    let pos = (p - d_pos.min(p)).min(m);
    // process the dense part
    let pro = p - d_pos - pos;

    
    println!("d_pos {d_pos:}");
    println!("d_neg {d_neg:}");
    // 
    println!("pre {pre:}");
    println!("pos {pos:}");
    println!("pro {pro:}");
    unsafe {
        if pos > 0 {
            // ltriangle::lmult_ut(xptr.add(pre), yptr, tptr, pre, pro, pos, m, p - pre, n, s_x, s_y, s_t);
            ltriangle::lmult_ut(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        } else {
            kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
