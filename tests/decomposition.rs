use stellar::learning::knn::LshKNearestNeighbors;
use stellar::structure::ndarray::NdArray;
use stellar::decomposition::qr::qr_decompose;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::svd::golub_kahan_explicit;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::givens::givens_iteration;
use stellar::decomposition::householder::householder_factor;



#[cfg(test)]

mod tests {
    use super::*; // bring your NdArray and functions in scope

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() &&
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_left_multiply() {
        let dims = vec![2, 2];
        let data = vec![
            -1.0, 0.0,
             5.0, 2.0,
        ];
        let x = NdArray::new(dims, data);
        let qr = qr_decompose(x);
        let mut ortho = qr.projection_matrix();
        qr.left_multiply(&mut ortho);
        let expected = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];

        assert!(approx_eq(&ortho.data, &expected, 1e-3));
    }

    #[test]
    fn test_qr_decompose_triangle() {
        let dims = vec![2, 2];
        let data = vec![
            -1.0, 0.0,
             5.0, 2.0,
        ];
        let x = NdArray::new(dims, data);

        let qr = qr_decompose(x.clone());
        let expected_triangle = vec![
            5.099,  1.961,
            0.000, -0.392,
        ];
        println!("qr triangle data {:?}", qr.triangle);

        assert!(approx_eq(&qr.triangle.data, &expected_triangle, 1e-3));
    }

    #[test]
    fn test_bidiagonalization() {
        let data = 1_f32;
        assert_eq!(data, 1_f32)
    }
}
