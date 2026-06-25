#[cfg(test)]
#[cfg(feature = "avx2")]
mod test_fma_behavior {
    use crate::algebra::bmethods::contractions::tensor_contraction;
    use crate::algebra::bmethods::interface::*;
    use crate::algebra::bmethods::tests::helpers::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::{generate_random_matrix, generate_random_vector};
    use crate::structure::ndarray::NdArray;

    #[test]
    fn test_fma_equivalence() {
        for (i, k, j) in test_data() {
            fma_matmul_equivalence(i, k, j);
            fma_tmatmul_equivalence(i, k, j);
        }
    }

    fn fma_matmul_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut t_d = generate_random_vector(m * n);
        let mut result = vec![0f32; m * n];
        let mut expected = basic_mult(&x, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        tensor_kernel(&x, &y, &mut t_d);
        let inspect = NdArray {
            dims: vec![m, n],
            data: t_d.clone(),
        };
        // println!("expected {expected:?}");
        // println!("actual {inspect:?}");
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn fma_tmatmul_equivalence(m: usize, p: usize, n: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        let mut t_d = generate_random_vector(m * n);
        x.transpose_inplace();
        // println!("x_base {x_base:?}");
        // println!("y {y:?}");
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        // let mut result = vec![0f32; m * n];
        // m, n, n b/c X is s tored in it's transposed state
        // tensor_tkernel(&x.data, &y.data, &mut result, m, p, n, m, n, n);
        tensor_tkernel(&x, &y, &mut t_d);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: t_d.clone(),
        };
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn fma_lower_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut x_base = x.clone();
        filter_lower_trapezoid(&mut x_base);
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        // tensor_lt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        tensor_lt_kernel(&x, &y, &mut result);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn fma_upper_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut x_base = x.clone();
        filter_upper_trapezoid(&mut x_base);
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        // tensor_ut_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        tensor_ut_kernel(&x, &y, &mut result);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn fma_rlower_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut y_base = y.clone();
        filter_lower_trapezoid(&mut y_base);
        // println!("x_base {x:?}");
        // println!("y_base {y_base:?}");
        let expected = basic_mult(&x, &y_base);
        let mut result = vec![0f32; m * n];
        // tensor_rlt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        tensor_rlt_kernel(&x, &y, &mut result);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        assert!(
            approx_vector_eq(&expected.data, &result[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn fma_rupper_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut y_base = y.clone();
        filter_upper_trapezoid(&mut y_base);
        // println!("x_base {x:?}");
        // println!("y_base {y_base:?}");
        let expected = basic_mult(&x, &y_base);
        let mut result = vec![0f32; m * n];
        // tensor_rut_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        tensor_rut_kernel(&x, &y, &mut result);
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
    fn fma_ltl_equivalence(m: usize, p: usize, n: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        filter_lower_trapezoid(&mut x_base);
        x.transpose_inplace(); // stored in transpose
        // println!("x_base {x_base:?}");
        // println!("y {y:?}");
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        // tensor_tlt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        // m, n, n b/c X is s tored in it's transposed state
        // tensor_tlt_block(&x.data, &y.data, &mut result, m, p, n, m, n, n);
        tensor_tlt_kernel(&x, &y, &mut result);
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
    fn fma_ltu_equivalence(m: usize, p: usize, n: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        // filter_lower_trapezoid(&mut x_base);
        filter_upper_trapezoid(&mut x_base);
        x.transpose_inplace(); // stored in transpose
        // println!("x_base {x_base:?}");
        // println!("y {y:?}");
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        // m, n, n b/c X is s tored in it's transposed state
        // tensor_tut_block(&x.data, &y.data, &mut result, m, p, n, m, n, n);
        tensor_tut_kernel(&x, &y, &mut result);
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
