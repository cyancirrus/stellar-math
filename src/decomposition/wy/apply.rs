use crate::algebra::bmethods::interface::{ stride_lt_kernel, stride_tlt_kernel, stride_tut_kernel, stride_ut_kernel};
use crate::decomposition::wy::primitives::import_slice;
// applies Lx;
#[inline(always)]
pub fn lhs_apply_l(
    l_yt: &[f32],
    q_argument: &[f32],
    t_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    acols: usize,
) {
    debug_assert!(t_buffer.len() >= rows * acols);
    stride_lt_kernel(
        l_yt, q_argument, t_buffer, 1, 0, rows, cols, acols, cols, acols, acols,
    );
}
/// the compact WY representation: `A = (I - Y T Y')X`.
#[rustfmt::skip]
#[inline(always)]
pub fn lhs_apply_q(
    l_yt: &[f32],
    tri: &[f32],
    x_argument: &[f32],
    t_buffer: &mut [f32],
    s_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    acols: usize,
) {
    debug_assert!(t_buffer.len() >= rows * acols);
    debug_assert!(s_buffer.len() >= rows * acols);
    debug_assert!(cols >= rows);
    stride_lhs_apply_q( l_yt, tri, x_argument, t_buffer, s_buffer, rows, cols, acols, cols, acols, acols);
}
/// the compact WY representation: `A = (I - Y T' Y')X`.
/// applies (I - YT'Y[thin]')x;
#[rustfmt::skip]
#[inline(always)]
pub fn lhs_apply_qt(
    l_yt: &[f32],
    tri: &[f32],
    x_argument: &[f32],
    t_buffer: &mut [f32],
    s_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    acols: usize,
) {
    debug_assert!(t_buffer.len() >= rows * acols);
    debug_assert!(s_buffer.len() >= rows * acols);
    debug_assert!(cols >= rows);
    stride_lhs_apply_qt(l_yt, tri, x_argument, t_buffer, s_buffer, rows, cols, acols, cols, acols, acols);
}
// applies Lx;
pub fn stride_lhs_apply_l(
    l_yt: &[f32],
    q_argument: &[f32],
    t_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    acols: usize,
    s_a: usize,
    s_x: usize,
    s_t: usize,
) {
    debug_assert!(t_buffer.len() >= rows * acols);
    stride_lt_kernel(
        l_yt, q_argument, t_buffer, 1, 0, rows, cols, acols, s_a, s_x, s_t,
    );
}
/// the compact WY representation: `A = (I - Y T Y')X`.
/// applies (I - Y[thin]TY')x;
#[rustfmt::skip]
pub fn stride_lhs_apply_q(
    l_yt: &[f32],
    tri: &[f32],
    x_argument: &[f32],
    t_buffer: &mut [f32],
    s_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    acols: usize,
    s_a:usize,
    s_x:usize,
    s_t:usize,
) {
    debug_assert!(t_buffer.len() >= rows * acols);
    debug_assert!(s_buffer.len() >= rows * acols);
    debug_assert!(cols >= rows);
    let s_tri = rows;
    import_slice(t_buffer, &x_argument[..rows * s_x]);

    // A = LX - YTY'X;
    // y'x
    stride_ut_kernel(
        &l_yt[1..],
        &x_argument[s_x..],
        t_buffer,
        0,
        0,
        rows,
        cols.saturating_sub(1),
        acols,
        s_a,
        s_x,
        s_t,
    );
    // t * [y'x];
    stride_lt_kernel(
        tri,
        t_buffer,
        s_buffer,
        1,
        0,
        rows,
        rows,
        acols,
        s_tri,
        s_t,
        s_t,
    );
    let mut toffset = 0;
    let mut xoffset = 0;
    for _ in 0..rows {
        for k in 0..acols {
            let v = -s_buffer[toffset + k];
            t_buffer[toffset + k] = v;
            s_buffer[toffset + k] = x_argument[xoffset + k] + v;
        }
        toffset += s_t;
        xoffset += s_x;
    }
    stride_tlt_kernel(
        &l_yt[1..],
        t_buffer,
        &mut s_buffer[acols..],
        rows - rows.min(cols) + 1,
        0,
        rows.saturating_sub(1),
        rows.saturating_sub(1),
        acols,
        s_a,
        s_t,
        s_t,
    );
}
/// the compact WY representation: `A = (I - Y T 'Y')X`.
/// applies (I - YT'Y[thin]')x;
#[rustfmt::skip]
pub fn stride_lhs_apply_qt(
    l_yt: &[f32],
    tri: &[f32],
    x_argument: &[f32],
    t_buffer: &mut [f32],
    s_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    acols: usize,
    s_a:usize,
    s_x:usize,
    s_t:usize,
) {
    debug_assert!(t_buffer.len() >= rows * acols);
    debug_assert!(s_buffer.len() >= rows * acols);
    debug_assert!(cols >= rows);
    let s_tri = rows;
    import_slice(t_buffer, &x_argument[..rows * s_t]);
    // A = LX - YTY'X;
    // y'x ; we can reduce work b/c now t is triangular is [t, 0];
    stride_ut_kernel(
        &l_yt[1..],
        &x_argument[s_t..],
        t_buffer,
        0,
        0,
        rows.saturating_sub(1),
        rows.saturating_sub(1),
        acols,
        s_a,
        s_x,
        s_t,
    );
    // t * [y'x];
    stride_tut_kernel(
        tri,
        t_buffer,
        s_buffer,
        0,
        0,
        // cols,
        rows,
        rows,
        acols,
        s_tri,
        s_t,
        s_t,
    );
    let mut toffset = 0;
    let mut xoffset = 0;
    for _ in 0..rows {
        for k in 0..acols {
            let v = -s_buffer[toffset + k];
            t_buffer[toffset + k] = v;
            s_buffer[toffset + k] = x_argument[xoffset + k] + v;
        }
        toffset += s_t;
        xoffset += s_x;
    }
    stride_tlt_kernel(
        &l_yt[1..],
        &t_buffer[..],
        &mut s_buffer[s_t..],
        1,
        0,
        cols.saturating_sub(1),
        rows,
        acols,
        s_a,
        s_t,
        s_t,
    );
}
