use stellar::structure::ndarray::NdArray;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::schur::real_schur;

#[cfg(test)]

mod schur_decomposition {
    use super::*; // bring your NdArray and functions in scope

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() &&
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }
    fn approx_scalar_eq(a:f32, b:f32, tol: f32) -> bool {
        (a-b).abs() < tol
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
            assert!(approx_scalar_eq(expected_eigens[i], schur.kernel.data[i * (dims[0] + 1)], 1e-3));

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
