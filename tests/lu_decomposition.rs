#[cfg(test)]
mod lu_decomposition {
    use rand::Rng;
    use rand_distr::StandardNormal;
    use stellar::algebra::ndmethods::create_identity_matrix;
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
    // functions
    fn reconstruction(x: NdArray) {
        let expected = x.clone();
        let lu = lu_decompose(x);
        let result = lu.reconstruct();
        assert!(approx_eq(&result.data, &expected.data))
    }
    fn generate_random(m:usize, n:usize) -> NdArray {
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

    fn reconstruction_random(n: usize) {
        reconstruction(generate_random(n, n))
    }

    fn test_left_apply_l(x:NdArray, y:NdArray) {
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
            reconstruction_random(n)
        }
    }
    #[test]
    fn random_left_apply_nxn_nxa() {
        let dimensions = vec![(1, 5), (2, 3), (7,7), (23,4)];
        for (n, a) in dimensions {
            let x = generate_random(n,n);
            let y = generate_random(n,a);
            test_left_apply_l(x, y) 
        }
    }
}
