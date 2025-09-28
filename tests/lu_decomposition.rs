#[cfg(test)]
mod lu_decomposition {
    use stellar::algebra::ndmethods::tensor_mult;
    use stellar::decomposition::lu::lu_decompose;
    use stellar::structure::ndarray::NdArray;
    use stellar::equality::approximate::approx_vector_eq;
    use stellar::random::generation::{generate_random_matrix, generate_random_vector};

    // functions
    fn reconstruction(x: NdArray) {
        let expected = x.clone();
        let lu = lu_decompose(x);
        let result = lu.reconstruct();
        assert!(approx_vector_eq(&result.data, &expected.data))
    }
    fn solve_inplace_vec_ax_y(a:NdArray, y:&mut [f32]) {
        let lu = lu_decompose(a);
        let expected = y.to_vec();
        let result = y;
        lu.solve_inplace_vec(result);
        lu.left_apply_u_vec( result);
        lu.left_apply_l_vec( result);
        assert!(approx_vector_eq(&expected, &result));

    }
    fn solve_inplace_ax_y(a:NdArray, y:&mut NdArray) {
        let lu = lu_decompose(a);
        let expected = y.clone();
        let result = y;
        lu.solve_inplace(result);
        lu.left_apply_u( result);
        lu.left_apply_l( result);
        assert!(approx_vector_eq(&expected.data, &result.data));

    }
    // matrix applications
    fn left_apply_l(x:NdArray, y:NdArray) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = lu_decompose(x);
        let mut l = lu.matrix.clone();
        let mut result = y.clone();
        for i in 0..rows {
            l.data[i * cols + i] = 1_f32;
            for j in i+1..cols {
                l.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &l, &y);
        lu.left_apply_l(&mut result);
        assert!(approx_vector_eq(&result.data, &expected.data))
    }
    fn left_apply_u(x:NdArray, y:NdArray) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = lu_decompose(x);
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
    fn right_apply_l(x:NdArray, y:NdArray) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = lu_decompose(x);
        let mut l = lu.matrix.clone();
        let mut result = y.clone();
        for i in 0..rows {
            l.data[i * cols + i] = 1_f32;
            for j in i+1..cols {
                l.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &y, &l);
        lu.right_apply_l(&mut result);
        assert!(approx_vector_eq(&result.data, &expected.data))
    }
    fn right_apply_u(x:NdArray, y:NdArray) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = lu_decompose(x);
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
    fn left_apply_l_vec(x:NdArray, y:Vec<f32>) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = lu_decompose(x);
        let mut l = lu.matrix.clone();
        let y_matrix = NdArray { dims: vec![cols, 1], data: y.clone()};
        let mut result = y.clone();
        for i in 0..rows {
            l.data[i * cols + i] = 1_f32;
            for j in i+1..cols {
                l.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &l, &y_matrix);
        lu.left_apply_l_vec(&mut result);
        assert!(approx_vector_eq(&result, &expected.data))
    }

    // vector applications
    fn left_apply_u_vec(x:NdArray, y:Vec<f32>) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = lu_decompose(x);
        let mut u = lu.matrix.clone();
        let y_matrix = NdArray { dims: vec![cols, 1], data: y.clone()};
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
    
    fn right_apply_l_vec(x:NdArray, y:Vec<f32>) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = lu_decompose(x);
        let mut l = lu.matrix.clone();
        let y_matrix = NdArray { dims: vec![1, rows], data: y.clone()};
        let mut result = y.clone();
        for i in 0..rows {
            l.data[i * cols + i] = 1_f32;
            for j in i+1..cols {
                l.data[i * cols + j] = 0_f32;
            }
        }
        let expected = tensor_mult(4, &y_matrix, &l);
        lu.right_apply_l_vec(&mut result);
        assert!(approx_vector_eq(&result, &expected.data))
    }
    fn right_apply_u_vec(x:NdArray, y:Vec<f32>) {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let lu = lu_decompose(x);
        let mut u = lu.matrix.clone();
        let y_matrix = NdArray { dims: vec![1, rows], data: y.clone()};
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
            reconstruction(generate_random_matrix(n,n))
        }
    }
    #[test]
    fn random_left_apply_l_nxn_nxa() {
        let dimensions = vec![(1, 5), (2, 3), (7,7), (23,4)];
        for (n, a) in dimensions {
            let x = generate_random_matrix(n,n);
            let y = generate_random_matrix(n,a);
            left_apply_l(x, y) 
        }
    }
    #[test]
    fn random_left_apply_u_nxn_nxa() {
        let dimensions = vec![(1, 5), (2, 3), (7,7), (23,4)];
        for (n, a) in dimensions {
            let x = generate_random_matrix(n,n);
            let y = generate_random_matrix(n,a);
            left_apply_u(x, y) 
        }
    }
    #[test]
    fn random_right_apply_l_axn_nxn() {
        let dimensions = vec![(1, 5), (2, 3), (7,7), (23,4)];
        for (n, a) in dimensions {
            let x = generate_random_matrix(n,n);
            let y = generate_random_matrix(a,n);
            right_apply_l(x, y) 
        }
    }
    #[test]
    fn random_right_apply_u_nxn_nxa() {
        let dimensions = vec![(1, 5), (2, 3), (7,7), (23,4)];
        for (n, a) in dimensions {
            let x = generate_random_matrix(n,n);
            let y = generate_random_matrix(a,n);
            right_apply_u(x, y) 
        }
    }
    #[test]
    fn random_left_apply_l_vec() {
        let dimensions = vec![ 2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_vector(n);
            left_apply_l_vec(x,y)
        }
    }
    #[test]
    fn random_left_apply_u_vec() {
        let dimensions = vec![ 2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_vector(n);
            left_apply_u_vec(x,y)
        }
    }
    #[test]
    fn random_right_apply_l_vec() {
        let dimensions = vec![ 2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_vector(n);
            right_apply_l_vec(x,y)
        }
    }
    #[test]
    fn random_right_apply_u_vec() {
        let dimensions = vec![ 2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let y = generate_random_vector(n);
            right_apply_u_vec(x,y)
        }
    }
    #[test]
    fn random_solve_inplace_vec_ax_y() {
        let dimensions = vec![ 2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let mut y = generate_random_vector(n);
            solve_inplace_vec_ax_y(x, &mut y)
        }
    }
    #[test]
    fn random_solve_inplace_ax_y() {
        let dimensions = vec![ 2, 3, 4, 7, 23];
        for n in dimensions {
            let x = generate_random_matrix(n, n);
            let mut y = generate_random_matrix(n, n);
            solve_inplace_ax_y(x, &mut y)
        }
    }
}
