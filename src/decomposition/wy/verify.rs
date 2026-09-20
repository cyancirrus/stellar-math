#[cfg(test)]
mod test_wy_reconstructions {
    use crate::algebra::ndmethods::create_identity_vector;
    use crate::algebra::ndmethods::matrix_mult;
    use crate::arch::SIMD_WIDTH;
    use crate::decomposition::wy::apply::{lhs_apply_l, lhs_apply_q, lhs_apply_qt};
    use crate::decomposition::wy::interface::{dense_solve, wy_decomposition};
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_vector;
    use crate::structure::ndarray::NdArray;

    /// A x ~= y, where x is the solution produced by dense_solve.
    fn check_wy_solve(rows: usize, cols: usize, tcols: usize) -> bool {
        debug_assert!(cols >= rows);
        let stride = cols;

        let mut l_yt = generate_random_vector(rows * cols);
        let original = NdArray {
            dims: vec![rows, cols],
            data: l_yt.clone(),
        };
        let mut tri = create_identity_vector(rows, rows);
        // kernel_forward_substitution wants tcols * SIMD_WIDTH
        let mut w = vec![0f32; (cols * SIMD_WIDTH).max(tcols * SIMD_WIDTH)];

        let mut x_argument = vec![0f32; cols * tcols];
        let y_argument = generate_random_vector(rows * tcols);
        // let mut t_buffer = vec![0f32; rows * tcols];
        // let mut s_buffer = vec![0f32; cols * tcols];
        let mut t_buffer = vec![0f32; (cols * tcols).max(tcols * SIMD_WIDTH)];
        let mut s_buffer = vec![0f32; (cols * tcols).max(tcols * SIMD_WIDTH)];

        wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, stride);
        dense_solve(
            &l_yt,
            &tri,
            &mut x_argument,
            &y_argument,
            &mut t_buffer,
            &mut s_buffer,
            rows,
            cols,
            tcols,
        );
        let solution = NdArray {
            dims: vec![cols, tcols],
            data: s_buffer[..cols * tcols].to_vec(),
        };
        let reconstruct = matrix_mult(&original, &solution);
        approx_vector_eq(
            &reconstruct.data[..rows * tcols],
            &y_argument[..rows * tcols],
        )
    }
    /// Q Q' x ~= x.
    fn check_wy_q_roundtrip(rows: usize, cols: usize, tcols: usize) -> bool {
        debug_assert!(cols >= rows);
        let stride = cols;

        let mut l_yt = generate_random_vector(rows * cols);
        let mut tri = create_identity_vector(rows, rows);
        let mut w = vec![0f32; cols * SIMD_WIDTH];

        let x_argument = generate_random_vector(rows * tcols);
        let mut t_buffer = vec![0f32; cols * tcols];
        let mut s_buffer = vec![0f32; cols * tcols];
        let mut w_buffer = vec![0f32; cols * tcols];

        wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, stride);
        lhs_apply_qt(
            &l_yt,
            &tri,
            &x_argument,
            &mut t_buffer,
            &mut s_buffer,
            rows,
            cols,
            tcols,
        );
        t_buffer.fill(0f32);
        lhs_apply_q(
            &l_yt,
            &tri,
            &s_buffer,
            &mut w_buffer,
            &mut t_buffer,
            rows,
            cols,
            tcols,
        );
        approx_vector_eq(&t_buffer[..rows * tcols], &x_argument[..rows * tcols])
    }
    /// L (Q x) ~= A x.
    fn check_wy_reconstruct(rows: usize, cols: usize, tcols: usize) -> bool {
        debug_assert!(cols >= rows);
        let stride = cols;

        let mut l_yt = generate_random_vector(rows * cols);
        let input = NdArray {
            dims: vec![rows, cols],
            data: l_yt.clone(),
        };
        let mut tri = create_identity_vector(rows, rows);
        let mut w = vec![0f32; cols * SIMD_WIDTH];

        let x_argument = generate_random_vector(cols * tcols);
        let arg_matrix = NdArray {
            dims: vec![cols, tcols],
            data: x_argument.clone(),
        };
        let mut t_buffer = vec![0f32; cols * tcols];
        let mut s_buffer = vec![0f32; cols * tcols];

        wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, stride);
        lhs_apply_q(
            &l_yt,
            &tri,
            &x_argument,
            &mut t_buffer,
            &mut s_buffer,
            rows,
            cols,
            tcols,
        );
        t_buffer.fill(0f32);
        lhs_apply_l(&l_yt, &s_buffer, &mut t_buffer, rows, cols, tcols);

        let expected = matrix_mult(&input, &arg_matrix);
        approx_vector_eq(&t_buffer[..rows * tcols], &expected.data[..rows * tcols])
    }
    #[test]
    fn test_wy_solve_square() {
        for dim in [2, 4, 7] {
            assert!(check_wy_solve(dim, dim, dim), "dim={dim}: Ax != y");
        }
    }
    #[test]
    fn test_wy_solve_wide() {
        for (rows, cols, tcols) in [(2, 4, 8), (4, 8, 12), (4, 4, 8), (14, 24, 24)] {
            assert!(
                check_wy_solve(rows, cols, tcols),
                "{rows}x{cols}x{tcols}: Ax != y"
            );
        }
    }
    #[test]
    fn test_wy_q_roundtrip() {
        for (rows, cols, tcols) in [(2, 8, 4), (4, 6, 8), (14, 24, 24), (24, 24, 24)] {
            assert!(
                check_wy_q_roundtrip(rows, cols, tcols),
                "{rows}x{cols}x{tcols}: QQ'x != x"
            );
        }
    }
    #[test]
    fn test_wy_reconstruct() {
        for (rows, cols, tcols) in [(2, 4, 4), (4, 6, 8), (8, 12, 6), (14, 24, 24)] {
            assert!(
                check_wy_reconstruct(rows, cols, tcols),
                "{rows}x{cols}x{tcols}: L(Qx) != Ax"
            );
        }
    }
    #[rustfmt::skip]
    #[test]
    fn test_wy_trials() {
        let trials = 10_000;
        let mut solve_failures = 0;
        let mut ortho_failures = 0;
        let mut recon_failures = 0;

        for _ in 0..trials {
            if !check_wy_solve(6, 6, 6) { solve_failures += 1; }
            if !check_wy_q_roundtrip(6, 6, 6) { ortho_failures += 1; }
            if !check_wy_reconstruct(6, 6, 6) { recon_failures += 1; }
        }

        println!("wy: {solve_failures} solve failures, {ortho_failures} orthogonality failures, {recon_failures} reconstruction failures / {trials}");
        assert!(solve_failures < 10, "too many solve failures: {solve_failures}");
        assert!(ortho_failures < 10, "too many orthogonality failures: {ortho_failures}");
        assert!(recon_failures < 10, "too many reconstruction failures: {recon_failures}");
    }
}
