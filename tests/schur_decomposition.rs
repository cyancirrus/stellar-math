#[cfg(test)]

mod schur_decomposition {
    use stellar::algebra::ndmethods::tensor_mult;
    use stellar::decomposition::schur::real_schur;
    use stellar::structure::ndarray::NdArray;
    const TOLERANCE: f32 = 1e-3;

    fn approx_vector_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| (x - y).abs() < TOLERANCE)
    }
    fn approx_scalar_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOLERANCE
    }

    #[test]
    fn test_schur_kernel() {
        let dims = vec![2, 2];
        let data = vec![-1.0, 0.0, 5.0, 2.0];
        let x = NdArray::new(dims.clone(), data);
        let schur = real_schur(x);
        let expected_eigens = vec![2.000, -1.000];
        for i in 0..dims[0] {
            assert!(approx_scalar_eq(
                expected_eigens[i],
                schur.kernel.data[i * (dims[0] + 1)]
            ));
        }
    }

    #[test]
    fn test_reconstruction() {
        let dims = vec![2, 2];
        let data = vec![-1.0, 0.0, 5.0, 2.0];
        let x = NdArray::new(dims.clone(), data.clone());
        let schur = real_schur(x);
        let q_star = schur.rotation.transpose();
        let q = &schur.rotation;

        let result = tensor_mult(4, &q_star, &schur.kernel);
        let result = tensor_mult(4, &result, q);
        assert!(approx_vector_eq(&result.data, &data,));
    }
    // #[test]
    // fn test_reconstruction_3() {
    //     let dims = vec![3, 3];
    //     let data = vec![
    //         -1.0, 0.0, 3.0,
    //          5.0, 2.0, 4.0,
    //          -3.0, 0.7, 1.2,
    //     ];
    //     let x = NdArray::new(dims.clone(), data.clone());
    //     let schur = real_schur(x);
    //     let q_star = schur.rotation.transpose();
    //     let q = &schur.rotation;

    //     let result = tensor_mult(4, &q_star, &schur.kernel);
    //     let result = tensor_mult(4, &result, q);
    //     println!("result {result:?}");
    //     assert!(approx_vector_eq(
    //         &result.data,
    //         &data,
    //     ));
    // }
}
