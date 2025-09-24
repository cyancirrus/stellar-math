#[cfg(test)]
mod qr_decomposition {
    use stellar::structure::ndarray::NdArray;
    use stellar::decomposition::qr::qr_decompose;
    use stellar::algebra::ndmethods::tensor_mult;
    use stellar::decomposition::schur::real_schur;
    const TOLERANCE:f32 = 1e-3;

    fn approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() &&
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < TOLERANCE)
    }
    fn approx_scalar_eq(a:f32, b:f32) -> bool {
        (a-b).abs() < TOLERANCE
    }
    
    #[test]
    fn test_reconstruction() {
        let dims = vec![2, 2];
        let data = vec![
            -1.0, 0.0,
             5.0, 2.0,
        ];
        let expected = data.clone();
        let x = NdArray::new(dims, data);
        let qr = qr_decompose(x);
        // let result = tensor_mult(4, &qr.projection_matrix(), &qr.triangle); 
        let mut result = qr.triangle.clone();
        qr.left_multiply(&mut result);
        assert!(approx_eq(&result.data, &expected));
    }

    #[test]
    fn test_zeroing_below_diagonal() {
        let dims = vec![2, 2];
        let data = vec![
            -1.0, 0.0,
             5.0, 2.0,
        ];
        let x = NdArray::new(dims, data);
        let qr = qr_decompose(x.clone());
        let ortho = qr.projection_matrix();
        let result = tensor_mult(4, &ortho, &x);
        for i in 1..x.dims[0] {
            for j in 0..i {
                assert!(approx_scalar_eq(0_f32, result.data[i*x.dims[1] + j]));

            }
        }
    }
    
    #[test]
    fn test_q_matrix_orthonormality() {
        let dims = vec![2, 2];
        let data = vec![
            -1.0, 0.0,
             5.0, 2.0,
        ];
        let x = NdArray::new(dims, data);
        let qr = qr_decompose(x);
        let ortho = qr.projection_matrix();
        let mut ortho_transpose = ortho.clone();
        ortho_transpose.transpose_square();
        let left_result = tensor_mult(4, &ortho, &ortho_transpose);
        let right_result = tensor_mult(4, &ortho_transpose, &ortho);
        let expected = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];

        assert!(approx_eq(&left_result.data, &expected));
        assert!(approx_eq(&right_result.data, &expected));
    }

    #[test]
    fn test_qr_projection_construction_and_implicit_mult_equivalence() {
        let dims = vec![2, 2];
        let data = vec![
            -1.0, 0.0,
             5.0, 2.0,
        ];
        let other = vec![
            1.5, -0.3,
             3.0, 1.2,
        ];
        let x = NdArray::new(dims.clone(), data);
        let mut y_implicit = NdArray::new(dims.clone(), other.clone());
        let y_explicit = NdArray::new(dims, other);
        let qr = qr_decompose(x);
        let projection = qr.projection_matrix();
        qr.left_multiply(&mut y_implicit);
        assert!(approx_eq(
            &tensor_mult(4, &projection, &y_explicit).data,
            &y_implicit.data,
        ));
    }

    #[test]
    fn test_qr_triangle() {
        // Assumes that:
        // 1. test_zeroing_below_diagonal passes
        // 2. test_q_matrix_orthonormality passes
        //
        // Only then does this triangle test make sense.
        let dims = vec![2, 2];
        let data = vec![
            -1.0, 0.0,
             5.0, 2.0,
        ];
        let x = NdArray::new(dims, data);
        let qr = qr_decompose(x);
        let expected = vec![
            5.099, 1.961,
            0.000, 0.392,
        ];
        assert!(approx_eq(&qr.triangle.data, &expected));
    }
}
