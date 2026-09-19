// #[cfg(test)]
// mod test_lq_reconstructions {
//     use crate::decomposition::wy::{
//         lhs_apply_l, lhs_apply_q, lhs_apply_qt, solve, wy_decomposition,
//     };
//     use crate::algebra::ndmethods::{create_identity_vector, matrix_mult};
//     use crate::equality::approximate::approx_vector_eq;
//     use crate::random::generation::generate_random_vector;
//     use crate::structure::ndarray::NdArray;

//     const ABSOLUTE: f32 = 1e-3; // Adjusted for f32 accumulation limits in medium-to-large dimensions

//     /// Validates that L * Q * X matches A * X for random inputs.
//     fn check_lq_reconstruction(rows: usize, cols: usize, acols: usize) -> bool {
//         debug_assert!(cols >= rows);

//         let mut l_yt = generate_random_vector(rows * cols);
//         let original_data = l_yt.clone();

//         let mut tri = create_identity_vector(rows, rows);
//         let mut w = vec![0f32; cols];

//         // 1. Compute WY Decomposition
//         wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, cols);

//         // 2. Generate random test matrix X (cols x acols)
//         let x_argument = generate_random_vector(cols * acols);

//         let mut t_buffer = vec![0f32; cols * acols];
//         let mut s_buffer = vec![0f32; cols * acols];

//         // 3. Apply Q (s_buffer = Q * X)
//         lhs_apply_q(
//             &l_yt,
//             &tri,
//             &x_argument,
//             &mut t_buffer,
//             &mut s_buffer,
//             rows,
//             cols,
//             acols,
//         );

//         // 4. Apply L (t_buffer = L * (Q * X))
//         t_buffer.fill(0f32);
//         lhs_apply_l(&l_yt, &s_buffer, &mut t_buffer, rows, cols, acols);

//         // 5. Reference calculation: A * X via direct matrix multiplication
//         let a_mat = NdArray {
//             dims: vec![rows, cols],
//             data: original_data,
//         };
//         let x_mat = NdArray {
//             dims: vec![cols, acols],
//             data: x_argument,
//         };
//         let expected = matrix_mult(&a_mat, &x_mat);

//         // Compare L*Q*X against A*X
//         approx_vector_eq(&t_buffer, &expected.data)
//     }

//     /// Validates that applying Q' and then Q recovers the original vector (Q'Q X == X).
//     fn check_q_orthogonality(rows: usize, cols: usize, acols: usize) -> bool {
//         debug_assert!(cols >= rows);

//         let mut l_yt = generate_random_vector(rows * cols);
//         let mut tri = create_identity_vector(rows, rows);
//         let mut w = vec![0f32; cols];

//         wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, cols);

//         let x_original = generate_random_vector(rows * acols);
//         let mut t_buffer = vec![0f32; cols * acols];
//         let mut s_buffer = vec![0f32; cols * acols];
//         let mut w_buffer = vec![0f32; rows * acols];

//         // Apply Q'
//         lhs_apply_qt(
//             &l_yt,
//             &tri,
//             &x_original,
//             &mut t_buffer,
//             &mut s_buffer,
//             rows,
//             cols,
//             acols,
//         );

//         // Apply Q back
//         t_buffer.fill(0f32);
//         lhs_apply_q(
//             &l_yt,
//             &tri,
//             &s_buffer,
//             &mut w_buffer,
//             &mut t_buffer,
//             rows,
//             cols,
//             acols,
//         );

//         approx_vector_eq(&t_buffer, &x_original)
//     }

//     #[test]
//     fn test_lq_recon_dimensions() {
//         // Test various valid shapes where cols >= rows
//         let test_cases = vec![
//             (2, 4, 2),
//             (4, 4, 4),  // Square
//             (4, 8, 4),  // Wide
//             (8, 16, 6), // Larger wide
//         ];

//         for (rows, cols, acols) in test_cases {
//             assert!(
//                 check_lq_reconstruction(rows, cols, acols),
//                 "LQ reconstruction failed for dims: rows={rows}, cols={cols}, acols={acols}"
//             );
//         }
//     }

//     #[test]
//     fn test_q_orthogonality_cases() {
//         let test_cases = vec![
//             (2, 4, 2),
//             (4, 6, 4),
//             (8, 12, 4),
//         ];

//         for (rows, cols, acols) in test_cases {
//             assert!(
//                 check_q_orthogonality(rows, cols, acols),
//                 "Q orthogonality (Q'Q = I) failed for dims: rows={rows}, cols={cols}"
//             );
//         }
//     }

//     #[rustfmt::skip]
//     #[test]
//     fn test_lq_statistical_trials() {
//         let trials = 1_000;
//         let mut recon_failures = 0;
//         let mut ortho_failures = 0;

//         for _ in 0..trials {
//             if !check_lq_reconstruction(4, 8, 4) {
//                 recon_failures += 1;
//             }
//             if !check_q_orthogonality(4, 8, 4) {
//                 ortho_failures += 1;
//             }
//         }

//         println!("LQ tests: {recon_failures} reconstruction failures, {ortho_failures} orthogonality failures / {trials}");
//         assert!(recon_failures < 5, "Too many LQ reconstruction failures: {recon_failures}");
//         assert!(ortho_failures < 5, "Too many Q orthogonality failures: {ortho_failures}");
//     }
// }
