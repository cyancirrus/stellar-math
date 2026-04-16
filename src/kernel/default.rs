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
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    block_v: usize,
    s_x: usize,
    s_y: usize,
) {
    // default rust method
    let mut xoffset = 0;
    let mut toffset = 0;
    let mut yoffset;
    for _i in 0..block_v {
        yoffset = 0;
        let x_row = &x[xoffset..xoffset + SIMD_WIDTH];
        for k in 0..SIMD_WIDTH {
            let scalar = x_row[k];
            let y_row = &y[yoffset..yoffset + SIMD_WIDTH];
            let t_row = &mut t[toffset..toffset + SIMD_WIDTH];
            for (t, y) in t_row.iter_mut().zip(y_row.iter()) {
                *t += scalar * y;
            }
            yoffset += s_y;
        }
        xoffset += s_x;
        toffset += s_y;
    }
}

#[inline(always)]
pub fn kernel_mult_scalar(
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    block_m: usize,
    block_k: usize,
    block_n: usize,
    s_x: usize,
    s_y: usize,
) {
    // simple method to handle edge cases
    let mut xoffset = 0;
    let mut toffset = 0;
    let mut yoffset;

    for _i in 0..block_m {
        yoffset = 0;
        let x_row = &x[xoffset..xoffset + block_k];
        for k in 0..block_k {
            let scalar = x_row[k];
            let y_row = &y[yoffset..yoffset + block_n];
            let t_row = &mut t[toffset..toffset + block_n];
            for (t, y) in t_row.iter_mut().zip(y_row.iter()) {
                *t += scalar * y;
            }
            yoffset += s_y;
        }
        xoffset += s_x;
        toffset += s_y;
    }
}
// pub fn kernel_mult_scalar(
//     a: &[f32],
//     b: &[f32],
//     c: &mut [f32],
//     block_m: usize,
//     block_k: usize,
//     block_n: usize,
//     s_x: usize,
//     s_y: usize,
// ) {
//     // simple method to handle edge cases
//     let mut aoffset = 0;
//     let mut coffset = 0;
//     let mut boffset;
//     for _i in 0..block_m {
//         boffset = 0;
//         // let a_row = &a[aoffset..aoffset + s_x];
//         let a_row = &a[aoffset..aoffset + block_k];
//         for k in 0..block_k {
//             let scalar = a_row[k];
//             let b_row = &b[boffset..boffset + block_n];
//             let c_row = &mut c[coffset..coffset + block_n];
//             for (c, b) in c_row.iter_mut().zip(b_row.iter()) {
//                 *c += scalar * b;
//             }
//             boffset += s_y;
//         }
//         // aoffset += s_x;
//         aoffset += s_x;
//         coffset += s_y;
//     }
// }
