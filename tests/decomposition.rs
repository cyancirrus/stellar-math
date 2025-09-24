use stellar::learning::knn::LshKNearestNeighbors;
use stellar::structure::ndarray::NdArray;
use stellar::decomposition::qr::qr_decompose;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::algebra::vector::vector_product;

use stellar::decomposition::svd::golub_kahan_explicit;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::givens::givens_iteration;



#[cfg(test)]

mod tests {
    use super::*; // bring your NdArray and functions in scope

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() &&
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }
    fn approx_scalar_eq(a:f32, b:f32, tol: f32) -> bool {
        (a-b).abs() < tol
    }
    #[test]
    fn test_ortho_zeroing() {
        let dims = vec![2, 2];
        let data = vec![
            -1.0, 0.0,
             5.0, 2.0,
        ];
        let x = NdArray::new(dims, data);
        let qr = qr_decompose(x.clone());
        println!("projections {:?}", qr.projections);
        let ortho = qr.projection_matrix();
        let result = tensor_mult(4, &ortho, &x);
        for i in 1..x.dims[0] {
            for j in 0..i {
                assert!(approx_scalar_eq(0_f32, result.data[i*x.dims[1] + j], 1e-3));

            }
        }
    }
    
    #[test]
    fn test_qr_q_orthogonality() {
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

        assert!(approx_eq(&left_result.data, &expected, 1e-3));
        assert!(approx_eq(&right_result.data, &expected, 1e-3));
    }

    #[test]
    fn test_qr_projection_and_implicit_mult() {
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
            1e-3
        ));
    }
    #[test]
    fn test_qr_decomp_equals_matrix() {
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
        assert!(approx_eq(&result.data, &expected, 1e-3));
    }

    #[test]
    fn test_qr_triangle() {
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
        assert!(approx_eq(&qr.triangle.data, &expected, 1e-3));
    }
    

    // #[test]
    // fn test_left_multiply() {
    //     let dims = vec![2, 2];
    //     let data = vec![
    //         -1.0, 0.0,
    //          5.0, 2.0,
    //     ];
    //     let x = NdArray::new(dims, data);
    //     let qr = qr_decompose(x);
    //     println!("here");

    //     let mut ortho = qr.projection_matrix();
    //     println!("pre transpose {ortho:?}");
    //     ortho.transpose_square();
    //     println!("post transpose {ortho:?}");
    //     qr.left_multiply(&mut ortho);
    //     println!("qr ortho data {:?}", ortho);
    //     let expected = vec![
    //         1.0, 0.0,
    //         0.0, 1.0,
    //     ];
    //     assert!(approx_eq(&ortho.data, &expected, 1e-3));
    // }

    // #[test]
    // fn test_qr_decompose_triangle() {
    //     let dims = vec![2, 2];
    //     let data = vec![
    //         -1.0, 0.0,
    //          5.0, 2.0,
    //     ];
    //     let x = NdArray::new(dims, data);

    //     let qr = qr_decompose(x.clone());
    //     let expected_triangle = vec![
    //         5.099,  1.961,
    //         0.000, -0.392,
    //     ];
    //     assert!(approx_eq(&qr.triangle.data, &expected_triangle, 1e-3));
    // }

    // // #[test]
    // // fn test_schur_kernel() {
    // //     let dims = vec![2, 2];
    // //     let data = vec![
    // //         -1.0, 0.0,
    // //          5.0, 2.0,
    // //     ];
    // //     let x = NdArray::new(dims, data);
    // //     let schur = real_schur(x);
    // //     let expected_kernel= vec![
    // //         2.000, -5.000,
    // //         0.000, -1.000,
    // //     ];
    // //     println!("Schur kernel {:?}", &schur.kernel);
    // //     assert!(approx_eq(&schur.kernel.data, &expected_kernel, 1e-3));

    // // }

    // // #[test]
    // // fn test_bidiagonalization() {
    // //     let data = 1_f32;
    // //     assert_eq!(data, 1_f32)
    // // }
}
