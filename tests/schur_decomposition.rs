#[cfg(test)]

mod schur_decomposition {
    use stellar::structure::ndarray::NdArray;
    use stellar::decomposition::schur::real_schur;
    const TOLERANCE:f32 = 1e-3;

    fn approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() &&
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < TOLERANCE)
    }
    fn approx_scalar_eq(a:f32, b:f32) -> bool {
        (a-b).abs() < TOLERANCE
    }

    #[test]
    fn test_schur_kernel() {
        let dims = vec![2, 2];
        let data = vec![
            -1.0, 0.0,
             5.0, 2.0,
        ];
        let x = NdArray::new(dims.clone(), data);
        let schur = real_schur(x);
        let expected_eigens= vec![ 2.000, -1.000];
        for i in 0..dims[0] {
            assert!(approx_scalar_eq(
                expected_eigens[i],
                schur.kernel.data[i * (dims[0] + 1)]
            ));
        }
    }
    
    #[test]
    fn test_reconstruction() {
        //TODO: Implement a right multiply
        // let dims = vec![2, 2];
        // let data = vec![
        //     -1.0, 0.0,
        //      5.0, 2.0,
        // ];
        // let x = NdArray::new(dims.clone(), data);
        // let schur = real_schur(x);
        // let expected_eigens= vec![ 2.000, -1.000];
        // for i in 0..dims[0] {
        //     assert!(approx_scalar_eq(expected_eigens[i], schur.kernel.data[i * (dims[0] + 1)], 1e-3));

        // }
    }
}
