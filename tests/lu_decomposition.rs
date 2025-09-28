#[cfg(test)]
mod lu_decomposition {
    use rand::Rng;
    use rand_distr::StandardNormal;
    use stellar::algebra::ndmethods::tensor_mult;
    use stellar::decomposition::lu::lu_decompose;
    use stellar::structure::ndarray::NdArray;

    // utilities
    const TOLERANCE: f32 = 1e-3;

    fn approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| (x - y).abs() < TOLERANCE)
    }
    fn approx_scalar_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOLERANCE
    }
    fn generate_random_matrix(m:usize, n:usize) -> NdArray {
        let mut rng = rand::rng();
        let mut data = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let val = rng.sample(StandardNormal);
                data[i * n + j] = val;
            }
        }
        NdArray {
            dims: vec![m, n],
            data,
        }
    }
    fn generate_random_vector(n:usize) -> Vec<f32> {
        let mut rng = rand::rng();
        let mut data = vec![0.0_f32;  n];
        for i in 0..n {
            data[i] = rng.sample(StandardNormal);
        }
        data
    }
    // functions
    fn reconstruction(x: NdArray) {
        let expected = x.clone();
        let lu = lu_decompose(x);
        let result = lu.reconstruct();
        assert!(approx_eq(&result.data, &expected.data))
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
        assert!(approx_eq(&result.data, &expected.data))
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
        assert!(approx_eq(&result.data, &expected.data))
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
        assert!(approx_eq(&result.data, &expected.data))
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
        assert!(approx_eq(&result.data, &expected.data))
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
        assert!(approx_eq(&result, &expected.data))
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
        println!("u {u:?}");
        println!("y {y:?}");
        let expected = tensor_mult(4, &u, &y_matrix);
        lu.left_apply_u_vec(&mut result);
        println!("result {result:?}");
        println!("..........");
        println!("expected {expected:?}");
        assert!(approx_eq(&result, &expected.data))
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
        assert!(approx_eq(&result, &expected.data))
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
        assert!(approx_eq(&result, &expected.data))
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
}
