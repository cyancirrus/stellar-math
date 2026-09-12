// ============================================================================
// LQ Block Decomposition — Compact WY Representation
// ============================================================================

// **`LqBlockDecomp`** — 0-alloc, in-place LQ factorization via blocked
// Householder reflections (compact WY form).
//
// - **Mutation contract:** Takes in a basis and mutates the original matrix
//   in place. On return, the lower-triangular part of the matrix holds `L`.
//
// - **Naming convention:**
//   - `Lower` ~ `L`
//   - `Householder` ~ `Y'`
//   - `Triangular` ~ `T`
//
// - **Storage layout:**
//   - `h`: packed storage `~ [ L \ Y' ]` — `L` and the Householder vectors
//     `Y'` share the same buffer, split across the diagonal.
//   - `t`: `T`, the block triangular factor for the compact WY update.
//
// - **Factorization form:** `A = LQ`, where `Q` is expressed implicitly via
//   the compact WY representation: `A = L * (I - Y T Y')`.
//
// *(Struct form below is kept only as a thin, allocating wrapper around the
// 0-alloc routine, for callers who want ownership instead of borrowing.)*
// pub struct LqBlockDecomp {
//     pub h: NdArray,
//     pub t: NdArray,
// }
use crate::algebra::bmethods::contractions::{
    tensor_contraction, tensor_lt_contraction, tensor_tlt_contraction, tensor_tut_contraction,
    tensor_ut_contraction,
};
use crate::algebra::bmethods::interface::{
    stride_kernel, stride_lt_kernel, stride_tlt_kernel, stride_tut_kernel, stride_ut_kernel,
};
const EPSILON: f32 = 1e-21;
/// params
///
/// takes in a slice, where we find the rotation vector
/// in order when multiplied by the original matrix returns
/// a zero'd matrix
fn params(v: &mut [f32]) -> f32 {
    let mut max_element = 0f32;
    for val in v.iter() {
        let v = val.abs();
        if v > max_element {
            max_element = v
        };
    }
    if max_element.abs() < EPSILON {
        return max_element;
    }
    let mut magnitude_squared = 0f32;
    let inv_max_element = 1f32 / max_element;
    for val in v.iter_mut() {
        *val *= inv_max_element;
        magnitude_squared += *val * *val;
    }
    let g = v[0].signum() * magnitude_squared.sqrt();
    let scale = v[0] + g;
    let inv_scale = 1f32 / scale;
    for val in v[1..].iter_mut() {
        *val *= inv_scale;
    }
    v[0] = -g * max_element;
    scale / g
}
/// triangle iteration
///
/// triangle iteration for WY decomposition of Q
/// LQ implementation which makes T ~ lower triangle
/// builds the kth row of the triangle matrix
///
/// * h: householder data stored in upper right matrix
/// * r: householder rotation for iteration k
/// * t: lower block traingular matrix growing row by row
/// * h_dim: col x col in original matrix space
/// * t_dim: row x row in original matrix space
/// * tau: scalar of similarity of the household reflection
/// * k: iteration index
fn triangle_iteration(
    h: &mut [f32],
    t: &mut [f32],
    r: &[f32],
    w: &mut [f32],
    h_dim: usize,
    t_dim: usize,
    k: usize,
    tau: f32,
) {
    // T[k] = ((T, 0), (-tau[k]* h[k]' Y[k-1]T[k-1], tau));
    // diagonal element stores the L[ii] element not householder
    let koffset = k * t_dim;
    t[koffset + k] = tau;
    if k == 0 {
        return;
    }

    let mut hoffset = 0;
    // h'Y :: Y
    w.fill(0f32);
    // stride_kernel(
        tensor_contraction(
        &h[k + 1..],
        r,
        w,
        k,
        // r.len(),
        h_dim.saturating_sub(k + 1),
        1,
        h_dim,
        1,
        1,
    );
    let (t_upper, t_target) = t.split_at_mut(koffset);
    for l in 0..k {
        w[l] = -tau * h[hoffset + k] - tau * w[l];
        hoffset += h_dim;
    }
    // stride_tut_kernel(
        tensor_tut_contraction(
        t_upper, w, t_target, 0, 0, 
        k, 
                                                        // t_dim.saturating_sub(1),
        k, 1, t_dim, 1, 1,
    );
}
pub fn wy_decomposition(
    l_yt: &mut [f32],
    t: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    stride: usize,
) {
    debug_assert!(rows <= cols);
    debug_assert!(rows <= w.len());
    t.fill(0f32);
    let mut active_range = rows;
    let mut offset = 0;
    for k in 0..rows {
        active_range -= 1;
        let (done_rows, todo_rows) = l_yt.split_at_mut(offset);
        let (curr_row, trail_rows) = todo_rows.split_at_mut(cols);

        let v_active = &mut curr_row[k..];
        let tau = params(v_active);
        // implicit 1f32 on the diagonal -> increment index and handle explicitly
        let v_tail = &v_active[1..];
        triangle_iteration(done_rows, t, v_tail, w, cols, rows, k, tau);

        let split_range = v_tail.len();
        w.fill(0f32);
        if active_range == 0 {
            return;
        }
        // stride_kernel(
            tensor_contraction(
            &trail_rows[k + 1..],
            v_tail,
            w,
            active_range,
            split_range,
            1,
            stride,
            1,
            1,
        );
        let mut roffset = k;
        for i in 0..active_range {
            w[i] = tau * w[i] + tau * trail_rows[roffset];
            trail_rows[roffset] -= w[i];
            for j in 0..split_range {
                trail_rows[roffset + j + 1] -= w[i] * v_tail[j];
            }
            roffset += stride;
        }
        offset += stride;
    }
}
/// copies memory from b into a
fn import_slice(target: &mut [f32], data: &[f32]) {
    target[..data.len()].copy_from_slice(data);
    target[data.len()..].fill(0f32);
}

pub fn lhs_apply_l(
    l_yt: &[f32],
    q_argument: &[f32],
    t_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    acols: usize,
    s_x: usize,
    s_t: usize,
) {
    debug_assert!(t_buffer.len() >= rows * acols);
    tensor_lt_contraction(
        l_yt, q_argument, t_buffer, 1, 0, rows, cols, acols, s_x, s_t, s_t,
    );
}
/// the compact WY representation: `A = (I - Y T Y')X`.
#[rustfmt::skip]
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
    let (s_x, s_t, s_tri) = (cols, acols, rows);
    import_slice(t_buffer, &x_argument[..rows * acols]);

    // A = LX - YTY'X;
    // y'x
    // stride_ut_kernel(
    tensor_ut_contraction(
        &l_yt[1..],
        &x_argument[s_t..],
        t_buffer,
        0,
        0,
        rows,
        cols.saturating_sub(1),
        acols,
        s_x,
        s_t,
        s_t,
    );
    // t * [y'x];
    // stride_lt_kernel(
    tensor_lt_contraction(
        tri,
        t_buffer,
        s_buffer,
        1,
        0,
        rows,
        // cols,
        rows,
        acols,
        s_tri,
        s_t,
        s_t,
    );
    for k in 0..rows * acols {
        let v = -s_buffer[k];
        t_buffer[k] = v;
        s_buffer[k] = x_argument[k] + v;
    }
    // stride_tlt_kernel(
    tensor_tlt_contraction(
        &l_yt[1..],
        &t_buffer[..],
        &mut s_buffer[acols..],
        rows - rows.min(cols) + 1,
        0,
        // cols.saturating_sub(1),
        rows.saturating_sub(1),
        rows.saturating_sub(1),
        acols,
        s_x,
        s_t,
        s_t,
    );
}
/// the compact WY representation: `A = (I - Y T' Y')X`.
/// applies (I - YTY[thin]')x;
#[rustfmt::skip]
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
    let (s_a, s_t, s_tri) = (cols, acols, rows);
    import_slice(t_buffer, &x_argument[..rows * acols]);
    // A = LX - YTY'X;
    // y'x ; we can reduce work b/c now t is triangular is [t, 0];
    // stride_ut_kernel(
    tensor_ut_contraction(
        &l_yt[1..],
        &x_argument[s_t..],
        t_buffer,
        0,
        0,
        rows.saturating_sub(1),
        rows.saturating_sub(1),
        acols,
        s_a,
        s_t,
        s_t,
    );
    // t * [y'x];
    // stride_tut_kernel(
    tensor_tut_contraction(
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
    for k in 0..rows * acols {
        let v = -s_buffer[k];
        t_buffer[k] = v;
        s_buffer[k] = x_argument[k] + v;
    }
    // stride_tlt_kernel(
    tensor_tlt_contraction(
        &l_yt[1..],
        &t_buffer[..],
        &mut s_buffer[acols..],
        // rows - rows.min(cols) + 1,
        1,
        0,
        // acols.saturating_sub(1),
        cols.saturating_sub(1),
        rows,
        acols,
        s_a,
        s_t,
        s_t,
    );
}
/// Solves Ax = y;
///  * l_yt : [l\y'] <- compressed mem storage form of WY(LQ)
///  * x    : the x in Ax=y for which we are solving can be a matrixvec
///  * y    : the y in Ax=y for which we are solving can be a matrixvec
///  * w    : workspace vector so canquickly scan sum per record
///  * s_a  : stride of the storage of l_yt ie A
///  * rows : number of rows in l_yt ie A
///  * tcols: target columns ie number of cols in x and in y
pub fn forward_solve(
    l_yt: &[f32],
    x: &mut [f32],
    y: &[f32],
    w: &mut [f32],
    s_a: usize,
    rows: usize,
    tcols: usize,
) {
    debug_assert!(w.len() >= tcols);
    debug_assert!(x.len() >= rows * tcols);
    for j in 0..tcols {
        // l00 * x_0j = y_0i
        x[j] = y[j] / l_yt[0];
    }
    let mut offset = s_a;
    let mut toffset = tcols;
    for i in 1..rows {
        for j in 0..tcols {
            w[j] = l_yt[offset] * x[j];
        }
        let mut koffset = tcols;
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
            x[toffset + j] = inv_scalar * (y[toffset + j] - w[j]);
        }
        offset += s_a;
        toffset += tcols;
    }
}
pub fn solve(
    l_yt: &[f32],
    tri: &[f32],
    x: &mut [f32],
    y: &[f32],
    t_buffer: &mut [f32],
    s_buffer: &mut [f32],
    s_a: usize,
    rows: usize,
    cols: usize,
    tcols: usize,
) {
    forward_solve(l_yt, x, y, t_buffer, s_a, rows, tcols);
    lhs_apply_qt(l_yt, tri, x, t_buffer, s_buffer, rows, cols, tcols);
}
