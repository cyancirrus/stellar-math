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
    tolerance: f32,
    absolute: f32,
) {
    if cols > rows {
        full_lbidiagonal(b, u, v, p, w, rows, cols, card, stride);
        if rows > 1 {
            full_decomp_lgivens(
                b, u, v, rows, cols, card, stride, max_iters, tolerance, absolute,
            );
        }
    } else {
        full_ubidiagonal(b, u, v, p, w, rows, cols, card, stride);
        if cols > 1 {
            full_decomp_ugivens(
                b, u, v, rows, cols, card, stride, max_iters, tolerance, absolute,
            );
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
    tolerance: f32,
    absolute: f32,
) {
    if cols > rows {
        lbidiagonal(b, p, w, rows, cols, card, stride);
        if rows > 1 {
            decomp_lgivens(b, card, stride, max_iters, tolerance, absolute);
        }
    } else {
        ubidiagonal(b, p, w, rows, cols, card, stride);
        if cols > 1 {
            decomp_ugivens(b, card, stride, max_iters, tolerance, absolute);
        }
    }
}

#[cfg(test)]
mod test_svd_diagonal_parity {
    use super::*;

    use crate::algebra::ndmethods::create_identity_vector;
    use crate::random::generation::generate_random_vector;

    const MAX_ITERS: usize = 40;
    const TOLERANCE: f32 = 1e-10;
    const ABSOLUTE: f32 = 1e-4;

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
            MAX_ITERS,
            TOLERANCE,
            ABSOLUTE,
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
            MAX_ITERS,
            TOLERANCE,
            ABSOLUTE,
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

#[cfg(test)]
mod test_svd_convergence_rate {
    use super::*;
    use crate::random::generation::generate_random_vector;
    const MAX_ITERS: usize = 40;
    const TOLERANCE: f32 = 1e-10;
    const ABSOLUTE: f32 = 1e-4;
    const CONVERGE_THRESHOLD: f32 = 1e-6;

    // off-diagonal energy just above the main diagonal (upper bidiagonal band)
    fn sum_upper_bidiagonal(m: &[f32], rows: usize, stride: usize) -> f32 {
        let mut error = 0f32;
        let mut offset = 1;
        for _ in 0..rows.saturating_sub(1) {
            error += m[offset].abs();
            offset += stride + 1;
        }
        error
    }

    // off-diagonal energy just below the main diagonal (lower bidiagonal band)
    fn sum_lower_bidiagonal(m: &[f32], rows: usize, stride: usize) -> f32 {
        let mut error = 0f32;
        let mut offset = stride;
        for _ in 0..rows.saturating_sub(1) {
            error += m[offset].abs();
            offset += stride + 1;
        }
        error
    }

    // total residual off-diagonal mass, whichever band is relevant post-convergence
    fn off_diagonal_residual(m: &[f32], rows: usize, stride: usize) -> f32 {
        sum_upper_bidiagonal(m, rows, stride) + sum_lower_bidiagonal(m, rows, stride)
    }

    fn run_convergence_trial(
        rows: usize,
        cols: usize,
        max_iters: usize,
        tol: f32,
        absolute: f32,
    ) -> f32 {
        let card = rows.min(cols);
        let stride = cols;
        let maximum = rows.max(cols);

        let mut b = generate_random_vector(rows * cols);
        let mut w = vec![0f32; maximum];
        let mut p = vec![0f32; maximum];

        svd_decomposition(
            &mut b, &mut p, &mut w, rows, cols, card, stride, max_iters, tol, absolute,
        );

        off_diagonal_residual(&b, card, stride)
    }
    fn convergence_rate_report(
        rows: usize,
        cols: usize,
        trials: usize,
        max_iters: usize,
        tol: f32,
        converge: f32,
        absolute: f32,
    ) -> Result<(), String> {
        let mut failures = 0usize;
        let mut max_residual = 0f32;
        let mut sum_residual = 0f64;
        let card = rows.min(cols);
        let convergence_threshold = (card as f32) * converge;
        let iterations = card * max_iters;

        for _ in 0..trials {
            let residual = run_convergence_trial(rows, cols, iterations, tol, absolute);
            sum_residual += residual as f64;
            if residual > max_residual {
                max_residual = residual;
            }
            if residual > convergence_threshold {
                failures += 1;
            }
        }

        let mean_residual = sum_residual / trials as f64;
        let rate = 100.0 * (trials - failures) as f64 / trials as f64;

        println!(
            "{rows}x{cols}: converged {rate:.3}% ({failures} failures / {trials}), \
             mean residual = {mean_residual:.3e}, max residual = {max_residual:.3e}"
        );

        if failures > 0 {
            Err(format!(
                "{rows}x{cols}: {failures}/{trials} trials failed to converge below {convergence_threshold:e}"
            ))
        } else {
            Ok(())
        }
    }
    #[test]
    fn test_convergence_rate_square_6x6() {
        convergence_rate_report(
            6,
            6,
            10_000,
            MAX_ITERS,
            TOLERANCE,
            CONVERGE_THRESHOLD,
            ABSOLUTE,
        )
        .unwrap();
    }

    #[test]
    fn test_convergence_rate_square_various() {
        let mut errors = Vec::new();
        for dim in [2, 3, 4, 5, 8] {
            if let Err(e) = convergence_rate_report(
                dim,
                dim,
                2_000,
                MAX_ITERS,
                TOLERANCE,
                CONVERGE_THRESHOLD,
                ABSOLUTE,
            ) {
                errors.push(e);
            }
        }
        assert!(errors.is_empty(), "\n{}", errors.join("\n"));
    }

    #[test]
    fn test_convergence_rate_tall() {
        let mut errors = Vec::new();
        for (rows, cols) in [(4, 2), (6, 4), (8, 4), (10, 6)] {
            if let Err(e) = convergence_rate_report(
                rows,
                cols,
                2_000,
                MAX_ITERS,
                TOLERANCE,
                CONVERGE_THRESHOLD,
                ABSOLUTE,
            ) {
                errors.push(e);
            }
        }
        assert!(errors.is_empty(), "\n{}", errors.join("\n"));
    }

    #[test]
    fn test_convergence_rate_wide() {
        let mut errors = Vec::new();
        for (rows, cols) in [(2, 4), (4, 6), (4, 8), (6, 10)] {
            if let Err(e) = convergence_rate_report(
                rows,
                cols,
                2_000,
                MAX_ITERS,
                TOLERANCE,
                CONVERGE_THRESHOLD,
                ABSOLUTE,
            ) {
                errors.push(e);
            }
        }
        assert!(errors.is_empty(), "\n{}", errors.join("\n"));
    }
}
