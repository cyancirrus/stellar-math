#[cfg(test)]
mod lower_upper {
    use stellar::algebra::ndmethods::tensor_mult;
    use stellar::decomposition::lower_upper::LuPivotDecompose;
    use stellar::equality::approximate::{approx_condition_eq, approx_vector_eq};
    use stellar::random::generation::{generate_random_matrix, generate_random_vector};
    use stellar::structure::ndarray::NdArray;

    // functions
    fn reconstruction(x: NdArray) {
        let expected = x.clone();
        let lu = LuPivotDecompose::new(x);
        let result = lu.reconstruct();
        assert!(approx_vector_eq(&result.data, &expected.data))
    }
    fn solve_inplace_vec_ax_y(a: NdArray, y: &mut [f32]) -> bool {
        // PA = LUx
        // Ax = y;
        // PAx = Py
        // LUx = Py;
        // => P'Py; when applying the compare
        let lu = LuPivotDecompose::new(a);
        let expected = y.to_vec();
        let result = y;
        let condition = lu.condition();
        lu.solve_inplace_vec(result);
        lu.left_apply_u_vec(result);
        lu.left_apply_l_vec(result);
        lu.unpivot_inplace_vec(result);
        approx_condition_eq(&expected, &result, &condition)
    }
    fn random_solve_inplace_vec_ax_y(dim: usize) -> bool {
        let x = generate_random_matrix(dim, dim);
        let mut y = generate_random_vector(dim);
        solve_inplace_vec_ax_y(x, &mut y)
    }
    fn solve_inplace_ax_y(a: NdArray, y: &mut NdArray) -> bool {
        // PA = LUx
        // Ax = y;
        // PAx = Py
        // LUx = Py;
        // => P'Py; when applying the compare
        let lu = LuPivotDecompose::new(a);
        let condition = lu.condition();
        let expected = y.data.clone();
        lu.solve_inplace(y);
        lu.left_apply_u(y);
        lu.left_apply_l(y);
        lu.unpivot_inplace(y);
        approx_condition_eq(&expected, &y.data, &condition)
    }
    fn random_solve_inplace_ax_y(dim: usize) -> bool {
        let x = generate_random_matrix(dim, dim);
        let mut y = generate_random_matrix(dim, dim);
        solve_inplace_ax_y(x, &mut y)
    }
    // matrix applications
    fn left_apply_l(x: NdArray, y: NdArray) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = LuPivotDecompose::new(x);
        let mut l = lu.matrix.clone();
        let mut result = y.clone();
        for i in 0..rows {
            l.data[i * cols + i] = 1_f32;
            for j in i + 1..cols {
                l.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &l, &y);
        lu.left_apply_l(&mut result);
        assert!(approx_vector_eq(&result.data, &expected.data))
    }
    fn left_apply_u(x: NdArray, y: NdArray) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = LuPivotDecompose::new(x);
        let mut u = lu.matrix.clone();
        let mut result = y.clone();
        for i in 1..rows {
            for j in 0..i {
                u.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &u, &y);
        lu.left_apply_u(&mut result);
        assert!(approx_vector_eq(&result.data, &expected.data))
    }
    fn right_apply_l(x: NdArray, y: NdArray) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = LuPivotDecompose::new(x);
        let mut l = lu.matrix.clone();
        let mut result = y.clone();
        for i in 0..rows {
            l.data[i * cols + i] = 1_f32;
            for j in i + 1..cols {
                l.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &y, &l);
        lu.right_apply_l(&mut result);
        assert!(approx_vector_eq(&result.data, &expected.data))
    }
    fn right_apply_u(x: NdArray, y: NdArray) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = LuPivotDecompose::new(x);
        let mut u = lu.matrix.clone();
        let mut result = y.clone();
        for i in 1..rows {
            for j in 0..i {
                u.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &y, &u);
        lu.right_apply_u(&mut result);
        assert!(approx_vector_eq(&result.data, &expected.data))
    }
    fn left_apply_l_vec(x: NdArray, y: Vec<f32>) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = LuPivotDecompose::new(x);
        let mut l = lu.matrix.clone();
        let y_matrix = NdArray {
            dims: vec![cols, 1],
            data: y.clone(),
        };
        let mut result = y.clone();
        for i in 0..rows {
            l.data[i * cols + i] = 1_f32;
            for j in i + 1..cols {
                l.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &l, &y_matrix);
        lu.left_apply_l_vec(&mut result);
        assert!(approx_vector_eq(&result, &expected.data))
    }

    // vector applications
    fn left_apply_u_vec(x: NdArray, y: Vec<f32>) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = LuPivotDecompose::new(x);
        let mut u = lu.matrix.clone();
        let y_matrix = NdArray {
            dims: vec![cols, 1],
            data: y.clone(),
        };
        let mut result = y.clone();
        for i in 1..rows {
            for j in 0..i {
                u.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &u, &y_matrix);
        lu.left_apply_u_vec(&mut result);
        assert!(approx_vector_eq(&result, &expected.data))
    }

    fn right_apply_l_vec(x: NdArray, y: Vec<f32>) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = LuPivotDecompose::new(x);
        let mut l = lu.matrix.clone();
        let y_matrix = NdArray {
            dims: vec![1, rows],
            data: y.clone(),
        };
        let mut result = y.clone();
        for i in 0..rows {
            l.data[i * cols + i] = 1_f32;
            for j in i + 1..cols {
                l.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &y_matrix, &l);
        lu.right_apply_l_vec(&mut result);
        assert!(approx_vector_eq(&result, &expected.data))
    }
    fn right_apply_u_vec(x: NdArray, y: Vec<f32>) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = LuPivotDecompose::new(x);
        let mut u = lu.matrix.clone();
        let y_matrix = NdArray {
            dims: vec![1, rows],
            data: y.clone(),
        };
        let mut result = y.clone();
        for i in 1..rows {
            for j in 0..i {
                u.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &y_matrix, &u);
        lu.right_apply_u_vec(&mut result);
        assert!(approx_vector_eq(&result, &expected.data))
    }

    // sample 2x2
    #[test]
    fn reconstruction_2x2() {
        let x = NdArray {
            dims: vec![2, 2],
            data: vec![-1.0, 0.0, 5.0, 2.0],
        };
        reconstruction(x)
    }
    #[test]
    fn reconstruction_3x3() {
        let x = NdArray {
            dims: vec![3, 3],
            data: vec![-1.0, 0.0, 3.0, 5.0, 2.0, 4.0, -3.0, 0.7, 1.2],
        };
        reconstruction(x)
    }
    #[test]
    fn reconstruction_random_nxn() {
        let dimensions = vec![1, 2, 5, 7, 23];
        for n in dimensions {
            reconstruction(generate_random_matrix(n, n))
        }
    }
    #[test]
    fn random_left_apply_l_nxn_nxa() {
        let dimensions = vec![(1, 5), (2, 3), (7, 7), (23, 4)];
        for (n, a) in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_matrix(n, a);
            left_apply_l(x, y)
        }
    }
    #[test]
    fn random_left_apply_u_nxn_nxa() {
        let dimensions = vec![(1, 5), (2, 3), (7, 7), (23, 4)];
        for (n, a) in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_matrix(n, a);
            left_apply_u(x, y)
        }
    }
    #[test]
    fn random_right_apply_l_axn_nxn() {
        let dimensions = vec![(1, 5), (2, 3), (7, 7), (23, 4)];
        for (n, a) in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_matrix(a, n);
            right_apply_l(x, y)
        }
    }
    #[test]
    fn random_right_apply_u_nxn_nxa() {
        let dimensions = vec![(1, 5), (2, 3), (7, 7), (23, 4)];
        for (n, a) in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_matrix(a, n);
            right_apply_u(x, y)
        }
    }
    #[test]
    fn random_left_apply_l_vec() {
        let dimensions = vec![2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_vector(n);
            left_apply_l_vec(x, y)
        }
    }
    #[test]
    fn random_left_apply_u_vec() {
        let dimensions = vec![2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_vector(n);
            left_apply_u_vec(x, y)
        }
    }
    #[test]
    fn random_right_apply_l_vec() {
        let dimensions = vec![2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_vector(n);
            right_apply_l_vec(x, y)
        }
    }
    #[test]
    fn random_right_apply_u_vec() {
        let dimensions = vec![2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_vector(n);
            right_apply_u_vec(x, y)
        }
    }
    #[test]
    fn test_vector_solve_accuracy() {
        let dims = vec![2, 3, 4, 7, 23];
        let k = dims.len();
        let n = 200;
        let den = k * n;
        let mut num = 0;
        for d in dims {
            for _ in 0..n {
                num += random_solve_inplace_vec_ax_y(d) as usize;
            }
        }
        // condition number is approximate
        assert!(num as f32 / den as f32 > 0.99);
    }
    #[test]
    fn test_matrix_solve_accuracy() {
        let dims = vec![2, 3, 4, 7, 23];
        let k = dims.len();
        let n = 200;
        let den = k * n;
        let mut num = 0;
        for d in dims {
            for _ in 0..n {
                num += random_solve_inplace_ax_y(d) as usize;
            }
        }
        // condition number is approximate
        assert!(num as f32 / den as f32 > 0.99);
    }
}
