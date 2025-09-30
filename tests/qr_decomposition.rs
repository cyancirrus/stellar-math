#[cfg(test)]
mod qr_decomposition {
    use stellar::algebra::ndmethods::create_identity_matrix;
    use stellar::algebra::ndmethods::tensor_mult;
    use stellar::decomposition::qr::qr_decompose;
    use stellar::structure::ndarray::NdArray;
    use stellar::random::generation::{generate_random_matrix};
    use stellar::equality::approximate::{approx_vector_eq, approx_scalar_eq};

    // test functions
    fn test_reconstruction(x: NdArray) {
        let expected = x.clone();
        let qr = qr_decompose(x);
        let mut result = qr.triangle.clone();
        qr.left_multiply(&mut result);
        assert!(approx_vector_eq(&result.data, &expected.data));
    }
    fn test_random(n: usize) {
        let matrix = generate_random_matrix(n, n);
        test_reconstruction(matrix)
    }
    fn test_zeroing_below_diagonal(x: NdArray) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let qr = qr_decompose(x.clone());
        let projection = qr.projection_matrix();
        let result = tensor_mult(4, &projection, &x);
        for i in 0..rows {
            for j in 0..i {
                assert!(approx_scalar_eq(result.data[i * cols + j], 0_f32));
            }
        }
    }
    fn test_orthogonal(x: NdArray) {
        let card = x.dims[0].min(x.dims[1]);
        let qr = qr_decompose(x);
        let ortho = qr.projection_matrix();
        let mut ortho_transpose = ortho.clone();
        ortho_transpose.transpose_square();
        let left_result = tensor_mult(4, &ortho, &ortho_transpose);
        let right_result = tensor_mult(4, &ortho_transpose, &ortho);
        let expected = create_identity_matrix(card);
        assert!(approx_vector_eq(&left_result.data, &expected.data));
        assert!(approx_vector_eq(&right_result.data, &expected.data));
    }
    fn test_triangle(x: NdArray, expected: NdArray) {
        let qr = qr_decompose(x);
        assert!(approx_vector_eq(&qr.triangle.data, &expected.data));
    }
    fn test_projection_and_implicit_mult_equivalence(x: NdArray, mut y_implicit: NdArray) {
        let qr = qr_decompose(x);
        let projection = qr.projection_matrix();
        let y_explicit = &tensor_mult(4, &projection, &y_implicit);
        qr.left_multiply(&mut y_implicit);
        assert!(approx_vector_eq(&y_implicit.data, &y_explicit.data));
    }
    // tests

    // sample 2x2
    #[test]
    fn test_reconstruction_2x2() {
        let x = NdArray {
            dims: vec![2, 2],
            data: vec![-1.0, 0.0, 5.0, 2.0],
        };
        test_reconstruction(x)
    }
    #[test]
    fn test_orthogonal_2x2() {
        let x = NdArray {
            dims: vec![2, 2],
            data: vec![-1.0, 0.0, 5.0, 2.0],
        };
        test_orthogonal(x)
    }
    #[test]
    fn test_zeroing_below_diagonal_2x2() {
        let x = NdArray {
            dims: vec![2, 2],
            data: vec![-1.0, 0.0, 5.0, 2.0],
        };
        test_zeroing_below_diagonal(x)
    }
    #[test]
    fn test_projection_and_implicit_mult_equivalence_2x2() {
        let x = NdArray {
            dims: vec![2, 2],
            data: vec![-1.0, 0.0, 5.0, 2.0],
        };
        let y = NdArray {
            dims: vec![2, 2],
            data: vec![1.5, -0.3, 3.0, 1.2],
        };
        test_projection_and_implicit_mult_equivalence(x, y)
    }
    #[test]
    fn test_triangle_2x2() {
        let x = NdArray {
            dims: vec![2, 2],
            data: vec![-1.0, 0.0, 5.0, 2.0],
        };
        let expected = NdArray {
            dims: vec![2, 2],
            data: vec![5.099, 1.961, 0.000, 0.392],
        };
        test_triangle(x, expected)
    }
    // sample 3X3
    #[test]
    fn test_reconstruction_3x3() {
        let x = NdArray {
            dims: vec![3, 3],
            data: vec![-1.0, 0.0, 3.0, 5.0, 2.0, 4.0, -3.0, 0.7, 1.2],
        };
        test_reconstruction(x)
    }
    #[test]
    fn test_orthogonal_3x3() {
        let x = NdArray {
            dims: vec![3, 3],
            data: vec![-1.0, 0.0, 3.0, 5.0, 2.0, 4.0, -3.0, 0.7, 1.2],
        };
        test_orthogonal(x)
    }
    #[test]
    fn test_zeroing_below_diagonal_3x3() {
        let x = NdArray {
            dims: vec![3, 3],
            data: vec![-1.0, 0.0, 3.0, 5.0, 2.0, 4.0, -3.0, 0.7, 1.2],
        };
        test_zeroing_below_diagonal(x)
    }
    #[test]
    fn test_random_nxn() {
        let numbers = vec![1, 2, 5, 7, 23];
        for n in numbers {
            test_random(n)
        }
    }
}
