#[cfg(test)]
mod test_kernel_block {
    use crate::algebra::bmethods::interface::*;
    use crate::algebra::bmethods::tests::helpers::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;
    // #[test]
    // fn test_contraction_equivalence() {
    //     for (i, k, j) in test_data() {
    //         test_contraction_equivalence_mkn(i, k, j);
    //     }
    // }
    #[test]
    fn test_gemm_equivalence() {
        for (i, k, j) in test_data() {
            println!("(i: {i:?}, k: {k:?}, j: {j:})");
            matmul_equivalence(i, k, j);
            tmatmul_equivalence(i, k, j);
        }
    }
    fn matmul_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut result = vec![0f32; m * n];
        let expected = basic_mult(&x, &y);
        // tensor_kernel(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        tensor_kernel(&x, &y, &mut result);
        // let inspect = NdArray {
        //     dims: vec![m, n],
        //     data: result.clone(),
        // };
        // println!("expected {expected:?}");
        // println!("actual {inspect:?}");
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn tmatmul_equivalence(m: usize, p: usize, n: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        x.transpose_inplace();
        // println!("x_base {x_base:?}");
        // println!("y {y:?}");
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        // m, n, n b/c X is s tored in it's transposed state
        // tensor_tkernel(&x.data, &y.data, &mut result, m, p, n, m, n, n);
        tensor_tkernel(&x, &y, &mut result);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        // println!("expected {expected:?}");
        // println!("actual {_inspect:?}");
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
}
