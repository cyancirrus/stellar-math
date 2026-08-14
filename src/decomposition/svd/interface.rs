use crate::decomposition::svd::bidiagonalization::{lbidiagonal, ubidiagonal};
use crate::decomposition::svd::bulge_chasing::{decomp_lgivens, decomp_ugivens};
#[rustfmt::skip]
use crate::decomposition::svd::verify::{
    full_ubidiagonal,
    full_lbidiagonal,
    full_decomp_ugivens,
    full_decomp_lgivens
};

pub fn full_svd_decomposition(
    b: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
    max_iters: usize,
    threshold: f32,
) {
    if cols > rows {
        full_lbidiagonal(b, u, v, p, w, rows, cols, card, stride);
        if rows > 1 {
            full_decomp_lgivens(b, u, v, rows, cols, card, stride, max_iters, threshold);
        }
    } else {
        full_ubidiagonal(b, u, v, p, w, rows, cols, card, stride);
        if cols > 1 {
            full_decomp_ugivens(b, u, v, rows, cols, card, stride, max_iters, threshold);
        }
    }
}
pub fn svd_decomposition(
    b: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
    max_iters: usize,
    threshold: f32,
) {
    if cols > rows {
        lbidiagonal(b, p, w, rows, cols, card, stride);
        if rows > 1 {
            decomp_lgivens(b, card, stride, max_iters, threshold);
        }
    } else {
        ubidiagonal(b, p, w, rows, cols, card, stride);
        if cols > 1 {
            decomp_ugivens(b, card, stride, max_iters, threshold);
        }
    }
}
#[cfg(test)]
mod test_svd_diagonal_parity {
    use super::*;

    use crate::algebra::ndmethods::create_identity_vector;
    use crate::random::generation::generate_random_vector;

    fn diagonal(b: &[f32], card: usize, stride: usize) -> Vec<f32> {
        (0..card).map(|i| b[i * stride + i]).collect()
    }

    fn check_diagonal_parity(rows: usize, cols: usize) -> bool {
        let card = rows.min(cols);
        let stride = cols;
        let maximum = rows.max(cols);

        let original = generate_random_vector(rows * cols);

        // full_svd_decomposition path
        let mut b_full = original.clone();
        let mut u = create_identity_vector(rows, rows);
        let mut v = create_identity_vector(cols, cols);
        let mut w_full = vec![0f32; maximum];
        let mut p_full = vec![0f32; maximum];
        full_svd_decomposition(
            &mut b_full,
            &mut u,
            &mut v,
            &mut p_full,
            &mut w_full,
            rows,
            cols,
            card,
            stride,
            40,
            1e-10,
        );

        // svd_decomposition path (no u/v accumulation)
        let mut b_bare = original.clone();
        let mut w_bare = vec![0f32; maximum];
        let mut p_bare = vec![0f32; maximum];
        svd_decomposition(
            &mut b_bare,
            &mut p_bare,
            &mut w_bare,
            rows,
            cols,
            card,
            stride,
            40,
            1e-10,
        );

        let diag_full = diagonal(&b_full, card, stride);
        let diag_bare = diagonal(&b_bare, card, stride);

        diag_full == diag_bare
    }

    #[test]
    fn test_diagonal_parity_square() {
        for dim in [2, 4, 7] {
            assert!(
                check_diagonal_parity(dim, dim),
                "dim={dim}: diagonals diverged"
            );
        }
    }

    #[test]
    fn test_diagonal_parity_wide() {
        for (rows, cols) in [(2, 4), (4, 6), (4, 8)] {
            assert!(
                check_diagonal_parity(rows, cols),
                "{rows}x{cols}: diagonals diverged"
            );
        }
    }

    #[test]
    fn test_diagonal_parity_tall() {
        for (rows, cols) in [(4, 2), (6, 4), (8, 4)] {
            assert!(
                check_diagonal_parity(rows, cols),
                "{rows}x{cols}: diagonals diverged"
            );
        }
    }

    #[rustfmt::skip]
    #[test]
    fn test_diagonal_parity_trials() {
        let trials = 10_000;
        let mut failures = 0;
        for _ in 0..trials {
            if !check_diagonal_parity(6, 6) { failures += 1; }
        }
        println!("diagonal parity: {failures} failures / {trials}");
        assert!(failures == 0, "exact-path diagonals diverged {failures} times — codepaths are not identical");
    }
}
