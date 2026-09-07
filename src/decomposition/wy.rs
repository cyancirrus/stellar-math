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
    tensor_lt_contraction, tensor_tlt_contraction,
    tensor_ut_contraction, tensor_tut_contraction
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
    let mut hoffset = 0;
    // h'Y :: Y
    let koffset = k * t_dim;
    let h_k_tail = r;
    for l in 0..k {
        // initial element of householder vector is 1
        let mut dot = h[hoffset + k];
        let h_i_tail = &h[hoffset + k + 1..hoffset + h_dim];
        for j in 0..h_k_tail.len() {
            dot += h_i_tail[j] * h_k_tail[j];
        }
        w[l] = dot;
        hoffset += h_dim;
    }
    let mut toffset = 0;
    let (t_upper, t_target) = t.split_at_mut(koffset);

    // h'T :: T ~ bottom-left triangular
    for l in 0..k {
        // outer product iteration style
        let outer = -w[l] * tau;
        let t_tail = &t_upper[toffset..=toffset + l];
        for j in 0..=l {
            t_target[j] += outer * t_tail[j];
        }
        toffset += t_dim;
    }
    t[koffset + k] = tau;
}
pub fn wy_decomposition(
    l_yt: &mut [f32],
    t: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    stride: usize,
) {
    // pub fn new(mut l_yt: NdArray, mut t_mat: NdArray, w: &mut [f32]) -> Self {
    // let (rows, cols) = (l_yt.dims[0], l_yt.dims[1]);
    debug_assert!(rows <= cols);
    debug_assert!(rows <= w.len());
    // let t = &mut t_mat.data;
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
        let mut roffset = 0;
        for _ in 0..active_range {
            let mut wi = trail_rows[roffset + k];
            {
                let mut targ_suffix = &mut trail_rows[roffset + k + 1..roffset + cols];
                targ_suffix = &mut targ_suffix[..split_range];
                for j in 0..split_range {
                    wi += targ_suffix[j] * v_tail[j];
                }
                wi *= tau;
                for j in 0..split_range {
                    targ_suffix[j] -= wi * v_tail[j];
                }
            }
            trail_rows[roffset + k] -= wi;
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
pub fn lhs_apply_q(
    l_yt: &[f32],
    tri: &[f32],
    x_argument: &mut [f32],
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
    tensor_lt_contraction(
        tri,
        t_buffer,
        s_buffer,
        1,
        0,
        rows,
        cols,
        acols,
        s_tri,
        s_t,
        s_t,
    );
    for k in 0..rows * acols {
        s_buffer[k] = -s_buffer[k];
        x_argument[k] += s_buffer[k];
    }
    tensor_tlt_contraction(
        &l_yt[1..],
        &s_buffer[..],
        &mut x_argument[acols..],
        rows - rows.min(cols) + 1,
        0,
        cols.saturating_sub(1),
        rows,
        acols,
        s_x,
        s_t,
        s_t,
    );
}
/// the compact WY representation: `A = (I - Y T' Y')X`.
pub fn lhs_apply_qt(
    l_yt: &[f32],
    tri: &[f32],
    x_argument: &mut [f32],
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
    tensor_tut_contraction(
        tri,
        t_buffer,
        s_buffer,
        
        0,
        0,
        cols,
        rows,
        acols,
        s_tri,
        s_t,
        s_t,
    );
    for k in 0..rows * acols {
        s_buffer[k] = -s_buffer[k];
        x_argument[k] += s_buffer[k];
    }
    tensor_tlt_contraction(
        &l_yt[1..],
        &s_buffer[..],
        &mut x_argument[acols..],
        rows - rows.min(cols) + 1,
        0,
        cols.saturating_sub(1),
        rows,
        acols,
        s_x,
        s_t,
        s_t,
    );
}
