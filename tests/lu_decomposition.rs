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
    fn test_reconstruction(x: NdArray) {
        let expected = x.clone();
        let lu = lu_decompose(x);
        let result = lu.reconstruct();
        println!("LU {:?}", lu.matrix); 
        // println!("L {:?}", lu.upper);
        // println!("u {:?}", lu.lower);
        // let result = tensor_mult(4, &lu.lower, &lu.upper);
        // println!("Result {result:?}");
        assert!(approx_eq(&result.data, &expected.data));
    }
    fn test_random(n: usize) {
        let mut rng = rand::rng();
        let mut data = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in i..n {
                let val = rng.sample(StandardNormal);
                data[i * n + j] = val;
                data[j * n + i] = val; // symmetric
            }
        }
        let matrix = NdArray {
            dims: vec![n, n],
            data,
        };
        test_reconstruction(matrix)
    }

    // sample 2x2
    #[test]
    fn test_reconstruction_2x2() {
        let x = NdArray {
            dims: vec![2, 2],
            data: vec![-1.0, 0.0, 5.0, 2.0],
        };
        test_reconstruction(x)
    }
    // #[test]
    // fn test_reconstruction_3x3() {
    //     let x = NdArray {
    //         dims: vec![3, 3],
    //         data: vec![-1.0, 0.0, 3.0, 5.0, 2.0, 4.0, -3.0, 0.7, 1.2],
    //     };
    //     test_reconstruction(x)
    // }
    // #[test]
    // fn test_random_nxn() {
    //     let numbers = vec![1, 2, 5, 7, 23];
    //     for n in numbers {
    //         test_random(n)
    //     }
    // }
}
