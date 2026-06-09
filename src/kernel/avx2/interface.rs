use crate::arch::SIMD_WIDTH;
use crate::kernel::avx2::{alligned, ltriangle, rtriangle, unalligned};
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
    let d_pos = d_add.saturating_sub(d_sub + 1);
    let d_neg = d_sub.saturating_sub(d_add);
    // pre-allign left boundary point
    let pre = d_pos;
    let pos = (n.saturating_sub(pre)).min(p);
    let pro = p.saturating_sub(pos);

    // let pre = d_pos ;
    // let pos = (n.saturating_sub(pre)).min(p );
    // let pro = p.saturating_sub(pos);
    println!("---------------------");
    println!("m: {m:}, p {p:}, n: {n:}");
    println!("---------------------");
    println!("d_pos {d_pos:?}, d_neg {d_neg:?}");
    println!("pre {pre:}, pro: {pro:}, pos: {pos:}");
    unsafe {
        // p = pre.saturating_sub(p);
        // pre = pre.min(p);
        // xptr = xptr.add(pre);
        // yptr = yptr.add(pre * s_y);
        // // index into specific column it's still outerproduct so same target
        // xptr = xptr.add(d_pos);
        // // index down for target row of y for outer product
        // yptr = yptr.add(d_pos * s_y);
        // // tptr is constant ie target output vectors are fixed
        // if pos != 0 {
        if pos + pre != 0 {
            println!("TRIANGLE");
            rtriangle::rmult_lt(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
        } else {
            // let m = 0;
            // let p = 1;
            // let n = 0;
            println!("DENSE");
            kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
        }
    }
}
// pub fn kernel_rlt_mult_simd(
//     mut xptr: *const f32,
//     mut yptr: *const f32,
//     tptr: *mut f32,
//     d_add: usize,
//     d_sub: usize,
//     m: usize,
//     mut p: usize,
//     n: usize,
//     s_x: usize,
//     s_y: usize,
//     s_t: usize,
// ) {
//     let d_pos = d_add.saturating_sub(d_sub + 1);
//     let d_neg = d_sub.saturating_sub(d_add);
//     // pre-allign left boundary point
//     // let pre = p.min(d_neg);
//     let pre = d_neg;
//     println!("pre {pre:}, d_pos {d_pos:}");
//     // handle triangle part of upper triangular
//     let pos = (n - n.min(d_pos)).min(p - pre);
//     // process the dense part
//     debug_assert!(d_pos + pos <= p, "d_pos: {d_pos}, pos:{pos}, p:{p}");
//     let pro = p - d_pos - pos;
//     unsafe {
//         if pos != 0 {
//             println!("mult, pre {pre:}, pro: {pro:}, pos: {pos:}");
//             rtriangle::rmult_lt(xptr, yptr, tptr, pre, pro, pos, m, p, n, s_x, s_y, s_t);
//         } else {
//             // let m = 0;
//             // let p = 1;
//             // let n = 0;
//             println!("m: {m:}, p {p:}, n: {n:}");
//             println!("dense");
//             kernel_mult_simd(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
//         }
//     }
// }
