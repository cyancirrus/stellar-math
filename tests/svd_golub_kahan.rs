#[cfg(test)]
mod svd_golub_kahan {
    use stellar::decomposition::svd::golub_kahan;
    use stellar::equality::approximate::{approx_scalar_eq, approx_vector_eq};
    use stellar::random::generation::generate_random_matrix;
    use stellar::structure::ndarray::NdArray;

    // test functions
    fn matrix_bidiagonalization(x: NdArray) {
        let mut error = 0f32;
        let bidiagonal = golub_kahan(x);
        let (rows, cols) = (bidiagonal.dims[0], bidiagonal.dims[1]);
        for i in 0..rows {
            for j in 0..cols {
                if i.max(j) - i.min(j) > 1 {
                    error += bidiagonal.data[i * cols + j];
                }
            }
        }
        println!("bidiagonal {bidiagonal:?}");
        assert!(approx_scalar_eq(error, 0_f32));
    }
    #[test]
    fn random_right_apply_u_vec() {
        let dimensions = vec![2, 3, 4, 7, 23];
        // let dimensions = vec![  4];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            matrix_bidiagonalization(x)
        }
    }
}
