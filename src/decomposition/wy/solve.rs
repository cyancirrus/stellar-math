use crate::algebra::bmethods::interface::stride_kernel;
use crate::arch::SIMD_WIDTH;

/// Solves Ax = y;
///  * l_yt : [l\y'] <- compressed mem storage form of WY(LQ)
///  * x    : the x in Ax=y for which we are solving can be a matrixvec
///  * y    : the y in Ax=y for which we are solving can be a matrixvec
///  * w    : workspace vector so canquickly scan sum per record
///  * rows : number of rows in l_yt ie A
///  * tcols: target columns ie number of cols in x and in y
///  * s_a  : stride of the storage of l_yt ie A
///  * s_x  : stride of the storage of x
///  * s_y  : stride of the storage of y
pub fn forward_solve(
    l_yt: &[f32],
    x: &mut [f32],
    y: &[f32],
    w: &mut [f32],
    rows: usize,
    tcols: usize,
    s_a: usize,
    s_x: usize,
    s_y: usize,
) {
    debug_assert!(w.len() >= tcols);
    debug_assert!(x.len() >= rows * s_x);
    for j in 0..tcols {
        // l00 * x_0j = y_0i
        x[j] = y[j] / l_yt[0];
    }
    let mut offset = s_a;
    let mut xoffset = s_x;
    let mut yoffset = s_y;
    for i in 1..rows {
        for j in 0..tcols {
            w[j] = l_yt[offset] * x[j];
        }
        let mut koffset = s_x;
        for k in 1..i {
            let scalar = l_yt[offset + k];
            for j in 0..tcols {
                w[j] += scalar * x[koffset + j];
            }
            koffset += tcols;
        }
        let inv_scalar = 1f32 / l_yt[offset + i];
        for j in 0..tcols {
            // dot + lii * x_i = y_i
            x[xoffset + j] = inv_scalar * (y[yoffset + j] - w[j]);
        }
        offset += s_a;
        xoffset += s_x;
        yoffset += s_y;
    }
}
/// Solves Ax = y;
///  * l_yt : [l\y'] <- compressed mem storage form of WY(LQ)
///  * x    : the x in Ax=y for which we are solving can be a matrixvec
///  * y    : the y in Ax=y for which we are solving can be a matrixvec
///  * w    : workspace vector so canquickly scan sum per record
///  * h    : workspace vector so canquickly scan sum per record
///  * rows : number of rows in l_yt ie A
///  * tcols: target columns ie number of cols in x and in y
///  * s_a  : stride of the storage of l_yt ie A
///  * s_x  : stride of the storage of x
///  * s_y  : stride of the storage of y
pub fn kernel_forward_solve(
    l_yt: &[f32],
    x: &mut [f32],
    y: &[f32],
    h: &mut [f32],
    rows: usize,
    tcols: usize,
    s_a: usize,
    s_x: usize,
    s_y: usize,
) {
    debug_assert!(h.len() >= tcols * SIMD_WIDTH);
    debug_assert!(x.len() >= rows * s_x);
    let mut offset = 0;
    let mut xoffset = 0;
    let mut yoffset = 0;
    let mut i = 0;
    let blocks = rows.div_ceil(SIMD_WIDTH);

    for _ in 0..blocks {
        let m = SIMD_WIDTH.min(rows - i);
        stride_kernel(&l_yt[offset..], x, h, m, i, tcols, s_a, s_x, s_y);
        let mut roffset = 0;
        let original = xoffset;

        for r in 0..m {
            let mut koffset = original;
            let w_i = &mut h[roffset..roffset + s_x];
            for k in 0..r {
                let scalar = l_yt[offset + i + k];
                for j in 0..tcols {
                    w_i[j] += scalar * x[koffset + j];
                }
                koffset += s_x;
            }
            let inv_scalar = 1f32 / l_yt[offset + i + r];
            for j in 0..tcols {
                // dot + lii * x_i = y_i
                x[xoffset + j] = inv_scalar * (y[yoffset + j] - w_i[j]);
                w_i[j] = 0f32;
            }
            offset += s_a;
            yoffset += s_y;
            xoffset += s_x;
            roffset += s_x;
        }
        i += SIMD_WIDTH;
    }
}
