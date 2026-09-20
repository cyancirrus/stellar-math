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
use crate::decomposition::wy::apply::stride_lhs_apply_qt;
use crate::decomposition::wy::factorization::kernel_wy_decomposition;
use crate::decomposition::wy::solve::forward_substitution;
use crate::decomposition::wy::solve::kernel_forward_substitution;

#[inline(always)]
pub fn wy_decomposition(
    l_yt: &mut [f32],
    t: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    stride: usize,
) {
    // could look at my linear form and block only if hits size threshold for kernel
    kernel_wy_decomposition(l_yt, t, w, rows, cols, stride);
}
#[inline(always)]
pub fn dense_solve(
    l_yt: &[f32],
    tri: &[f32],
    x: &mut [f32],
    y: &[f32],
    t_buffer: &mut [f32],
    s_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    tcols: usize,
) {
    // if rows >> 3 == 0 && cols >> 3 == 0 && tcols >> 3 == 0 {
    //     stride_solve(
    //         l_yt, tri, x, y, t_buffer, s_buffer, rows, cols, tcols, cols, tcols, tcols,
    //     );
    // } else {
    kernel_solve(
        l_yt, tri, x, y, t_buffer, s_buffer, rows, cols, tcols, cols, tcols, tcols,
    );
    // }
}
#[inline(always)]
pub fn kernel_solve(
    l_yt: &[f32],
    tri: &[f32],
    x: &mut [f32],
    y: &[f32],
    t_buffer: &mut [f32],
    s_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    tcols: usize,
    s_a: usize,
    s_x: usize,
    s_t: usize,
) {
    kernel_forward_substitution(l_yt, x, y, t_buffer, rows, tcols, s_a, s_x, s_t);
    stride_lhs_apply_qt(
        l_yt, tri, x, t_buffer, s_buffer, rows, cols, tcols, s_a, s_x, s_t,
    );
}
#[inline(always)]
pub fn stride_solve(
    l_yt: &[f32],
    tri: &[f32],
    x: &mut [f32],
    y: &[f32],
    t_buffer: &mut [f32],
    s_buffer: &mut [f32],
    rows: usize,
    cols: usize,
    tcols: usize,
    s_a: usize,
    s_x: usize,
    s_t: usize,
) {
    forward_substitution(l_yt, x, y, t_buffer, rows, tcols, s_a, s_x, s_t);
    stride_lhs_apply_qt(
        l_yt, tri, x, t_buffer, s_buffer, rows, cols, tcols, s_a, s_x, s_t,
    );
}
