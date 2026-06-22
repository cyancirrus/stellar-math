use crate::arch::SIMD_WIDTH;
use crate::kernel::avx2::{alligned, ltriangle, rtriangle, unalligned};
// #[inline(always)]
#[inline]
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
    d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    let d_pos = d_add.saturating_sub(d_sub);
    let d_neg = d_sub.saturating_sub(d_add);
    // preprocess the diagonal when on boundary
    let pre = d_neg;
    // process the diagonal when occurred in range
    let pro = d_pos;
    // process the dense part
    debug_assert!(m >= d_neg, "m {m:}, d_neg: {d_neg:}");
    let pos = (p - p.min(d_pos)).min(m - d_neg);
    unsafe {
        if pos != 0 {
            ltriangle::lmult_lt(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        } else {
            kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
pub fn kernel_tlt_mult_simd(
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
    // TODO: no transpose implementation yet
    let d_pos = d_add.saturating_sub(d_sub);
    let d_neg = d_sub.saturating_sub(d_add);
    // preprocess the diagonal when on boundary
    let pre = d_neg;
    // process the diagonal when occurred in range
    let pro = d_pos.min(p);
    // process the dense part
    // debug_assert!(m >= d_neg, "m {m:}, d_neg: {d_neg:}");
    let pos = (p - p.min(d_pos)).min(m - d_neg);
    unsafe {
        // if pos != 0 {
        ltriangle::lmult_tlt(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        // } else {
        //     kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        // }
    }
}
pub fn kernel_ut_mult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    d_add: usize,
    d_sub: usize,
    m: usize,
    mut p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    let d_pos = d_add.saturating_sub(d_sub);
    let d_neg = d_sub.saturating_sub(d_add);
    // pre-allign left boundary point
    let pre = m.min(d_neg);
    // handle triangle part of upper triangular
    let pos = (p - p.min(d_pos)).min(m - pre);
    // process the dense part
    debug_assert!(d_pos + pos <= p, "d_pos: {d_pos}, pos:{pos}, p:{p}");
    let pro = p - d_pos - pos;
    unsafe {
        p = p - d_pos;
        // index into specific column it's still outerproduct so same target
        xptr = xptr.add(d_pos);
        // index down for target row of y for outer product
        yptr = yptr.add(d_pos * s_y);
        // tptr is constant ie target output vectors are fixed
        if pos != 0 {
            ltriangle::lmult_ut(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        } else {
            kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
pub fn kernel_tut_mult_simd(
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
    let d_pos = d_add.saturating_sub(d_sub);
    let d_neg = d_sub.saturating_sub(d_add);
    // pre-allign left boundary point
    let pre = m.min(d_neg);
    // handle triangle part of upper triangular
    let pos = (p - p.min(d_pos)).min(m - pre);

    let pro = p.saturating_sub(d_pos + pos);
    // println!("d_pos {d_pos:}, d_neg {d_neg:}");
    println!("pre {pre:}, pro {pro:}, pos: {pos:}");
    unsafe {
        // if pos != 0 {
        ltriangle::lmult_tut(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        // } else {
        //     kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        // }
    }
}
/// handle when
/// 0 0 0
/// 0 0 0
/// * 0 0
/// * * 0
/// * * *
pub fn kernel_rlt_mult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    d_add: usize,
    d_sub: usize,
    m: usize,
    mut p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    let d_pos = d_add.saturating_sub(d_sub);
    let d_neg = (d_sub).saturating_sub(d_add);
    // pre how much the diagonal is shifted up and left
    let pre = d_pos;
    // how much triangle processing to be done
    let pos = (n.saturating_sub(pre)).min(p);
    // how much dense processes to perform
    let pro = p.saturating_sub(pos + d_neg);
    unsafe {
        xptr = xptr.add(d_neg);
        yptr = yptr.add(d_neg * s_y);
        p = p.saturating_sub(d_neg);
        if pos != 0 {
            rtriangle::rmult_lt(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        } else {
            kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
/// handle when
/// case 1
/// * * *
/// * * *
/// 0 * *
/// 0 0 *
/// case 2
/// * * * * *
/// 0 * * * *
/// 0 0 * * *
pub fn kernel_rut_mult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    d_add: usize,
    d_sub: usize,
    m: usize,
    mut p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    let d_pos = d_add.saturating_sub(d_sub);
    let d_neg = (d_sub).saturating_sub(d_add);
    // pre how much the diagonal is shifted up and left
    let pre = d_neg;
    // how much dense processes to perform
    let pro = d_pos.min(p);
    // how much triangle processing to be done
    let pos = (n.saturating_sub(pre)).min(p - pro);
    unsafe {
        if pos != 0 {
            rtriangle::rmult_ut(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        } else {
            kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
