#[cfg(test)]

mod schur_decomposition {
    use stellar::algebra::ndmethods::basic_mult;
    use stellar::algebra::ndmethods::create_identity_matrix;
    use stellar::decomposition::schur::real_schur;
    use stellar::equality::approximate::{
        approx_scalar_eq, approx_vector_eq, approx_vector_tol_eq,
    };
    use stellar::random::generation::generate_random_matrix;
    use stellar::structure::ndarray::NdArray;

    const TOLERANCE: f32 = 1e-3;

    #[test]
    fn test_schur_kernel() {
        let dims = vec![2, 2];
        let data = vec![-1.0, 0.0, 5.0, 2.0];
        let x = NdArray::new(dims.clone(), data);
        let schur = real_schur(x);
        let expected_eigens = [2.000, -1.000];
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
        let schur = real_schur(x.clone());
        let q = &schur.rotation;
        let q_star = &schur.rotation.transpose();

        let expect = basic_mult(&x, q);
        let expect = basic_mult(q_star, &expect);
        let result = schur.kernel;
        println!("expect {expect:?}");
        println!("result {result:?}");
        assert!(approx_vector_tol_eq(&expect.data, &result.data, TOLERANCE));
    }
    // #[test]
    // fn test_reconstruction() {
    //     let dims = vec![2, 2];
    //     let data = vec![-1.0, 0.0, 5.0, 2.0];
    //     let x = NdArray::new(dims.clone(), data.clone());
    //     let expect = x.clone();
    //     let schur = real_schur(x);
    //     // let q_star = schur.rotation.transpose();
    //     // let q = &schur.rotation;

    //     let result = basic_mult(&q_star, &schur.kernel);
    //     let result = basic_mult(&result, q);
    //     let result = basic_mult(&schur.rotation, &schur.kernel);
    //     // result.transpose_inplace();
    //     println!("expect {expect:?}");
    //     println!("result {result:?}");
    //     assert!(approx_vector_eq(&result.data, &data,));
    // }
    #[test]
    fn test_orthogonal() {
        for n in 1..8 {
            check_orthogonal(n);
        }
    }
    fn check_orthogonal(n: usize) {
        let x = generate_random_matrix(n, n);
        let schur = real_schur(x);
        let q = schur.rotation;
        let q_star = q.transpose();
        println!("q {q:?}");
        let expect = create_identity_matrix(n);
        let result = basic_mult(&q, &q_star);
        println!("expect {expect:?}");
        println!("result {result:?}");
        assert!(approx_vector_eq(&result.data, &expect.data,));
    }
    #[test]
    fn test_reconstruction_3() {
        let dims = vec![3, 3];
        let data = vec![-1.0, 2.0, 3.0, 5.0, 2.0, 4.0, -3.0, 0.7, 1.2];
        let x = NdArray::new(dims.clone(), data.clone());
        let schur = real_schur(x.clone());
        // let q = &schur.rotation;
        // let q_star = &schur.rotation.transpose();
        let q = &schur.rotation;
        let q_star = &schur.rotation.transpose();

        let expect = basic_mult(&x, q);
        let expect = basic_mult(q_star, &expect);
        // let result = basic_mult(&schur.kernel, &schur.rotation);
        let result = schur.kernel;
        println!("expect {expect:?}");
        println!("result {result:?}");
        assert!(approx_vector_tol_eq(&expect.data, &result.data, TOLERANCE));
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

    //     let result = basic_mult(&q_star, &schur.kernel);
    //     let result = basic_mult(&result, q);
    //     println!("result {result:?}");
    //     assert!(approx_vector_eq(
    //         &result.data,
    //         &data,
    //     ));
    // }
}
