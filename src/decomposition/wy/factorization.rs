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
use crate::algebra::bmethods::interface::{stride_kernel, stride_tut_kernel};
use crate::decomposition::wy::primitives::params;
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
    // Y'h :: Y
    // w.fill(0f32);
    stride_kernel(
        &h[k + 1..],
        r,
        w,
        k,
        h_dim.saturating_sub(k + 1),
        1,
        h_dim,
        1,
        1,
    );
    // -tau * X; // currently don't have scalar capabilities in kernel
    let (t_upper, t_target) = t.split_at_mut(koffset);
    for l in 0..k {
        w[l] = -tau * h[hoffset + k] - tau * w[l];
        hoffset += h_dim;
    }
    stride_tut_kernel(t_upper, w, t_target, 0, 0, k, k, 1, t_dim, 1, 1);
}
pub fn kernel_wy_decomposition(
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
        stride_kernel(
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
        // X - tau * UU'X; // currently don't have scalar capabilities in kernel
        let mut roffset = k;
        for i in 0..active_range {
            w[i] = tau * w[i] + tau * trail_rows[roffset];
            trail_rows[roffset] -= w[i];
            for j in 0..split_range {
                trail_rows[roffset + j + 1] -= w[i] * v_tail[j];
            }
            roffset += stride;
            w[i] = 0f32;
        }
        offset += stride;
    }
}
