#[cfg(test)]
#[cfg(feature = "avx2")]
mod test_stride_behavior {
    use crate::algebra::bmethods::interface::*;
    use crate::algebra::bmethods::tests::helpers::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::{approx_stride_eq, approx_vector_eq};
    use crate::random::generation::{generate_random_matrix, generate_random_vector};
    use crate::structure::ndarray::NdArray;

    #[test]
    fn test_sfma_equivalence() {
        let ds_x = 4;
        let ds_y = 8;
        for (i, k, j) in test_data() {
            let s_x = ds_x + k;
            let s_y = ds_y + j;
            let s_t = j;
            // sfma_matmul_equivalence(i, k, j, s_x, s_y, s_t);
            // sfma_tmatmul_equivalence(i, k, j, s_x, s_y, s_t);
            // sfma_lower_equivalence(i, k, j, s_x, s_y, s_t);
            // sfma_upper_equivalence(i, k, j, s_x, s_y, s_t);
            // sfma_rlower_equivalence(i, k, j, s_x, s_y, s_t);
            // sfma_rupper_equivalence(i, k, j, s_x, s_y, s_t);
            // sfma_ltl_equivalence(i, k, j, s_x, s_y, s_t);
            // sfma_ltu_equivalence(i, k, j, s_x, s_y, s_t);
            // sfma_ltl_equivalence(i, k, j, s_x, s_y, s_t);
        }
    }
    fn sfma_matmul_equivalence(m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut t_d = generate_random_vector(m * n);
        let mut expected = basic_mult(&x, &y);

        let mut x_stride = vec![0f32; m * s_x];
        let mut y_stride = vec![0f32; p * s_y];
        increment(&mut expected.data, &t_d, m, n, n, n);

        pack_stride(&mut x_stride, &x.data, m, p, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        stride_kernel(&x_stride, &y_stride, &mut t_d, m, p, n, s_x, s_y, s_t);
        // tensor_kernel(&x_stride, &y_stride, &mut t_d);
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_tmatmul_equivalence(m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
        // m, n, n b/c X is s tored in it's transposed state
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let x_base = x.clone();
        let mut t_d = generate_random_vector(m * n);
        x.transpose_inplace();
        let mut x_stride = vec![0f32; m * s_x];
        let mut y_stride = vec![0f32; p * s_y];
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        pack_stride(&mut x_stride, &x.data, p, m, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        stride_tkernel(&x.data, &y.data, &mut t_d, m, p, n, s_x, s_y, s_t);
        // tensor_tkernel(&x, &y, &mut t_d);
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_lower_equivalence(m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut x_stride = vec![0f32; m * s_x];
        let mut y_stride = vec![0f32; p * s_y];
        let (d_add, d_sub) = (0, 0);
        pack_stride(&mut x_stride, &x.data, m, p, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        let mut t_d = generate_random_vector(m * n);
        let mut x_base = x.clone();
        filter_lower_trapezoid(&mut x_base);
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        stride_lt_kernel(
            &x.data, &y.data, &mut t_d, d_add, d_sub, m, p, n, s_x, s_y, s_t,
        );
        // stride_lt_kernel(&x.data, &y.data, &mut t_d, m, p, n, s_x, s_y, s_t);
        // tensor_lt_kernel(&x, &y, &mut t_d);
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_upper_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut t_d = generate_random_vector(m * n);
        let mut x_base = x.clone();
        filter_upper_trapezoid(&mut x_base);
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        // tensor_ut_block(&x.data, &y.data, &mut t_d, m, p, n, p, n, n);
        tensor_ut_kernel(&x, &y, &mut t_d);
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_rlower_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut t_d = generate_random_vector(m * n);
        let mut y_base = y.clone();
        filter_lower_trapezoid(&mut y_base);
        let mut expected = basic_mult(&x, &y_base);
        increment(&mut expected.data, &t_d, m, n, n, n);
        // tensor_rlt_block(&x.data, &y.data, &mut t_d, m, p, n, p, n, n);
        tensor_rlt_kernel(&x, &y, &mut t_d);
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_rupper_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut t_d = generate_random_vector(m * n);
        let mut y_base = y.clone();
        filter_upper_trapezoid(&mut y_base);
        let mut expected = basic_mult(&x, &y_base);
        increment(&mut expected.data, &t_d, m, n, n, n);
        // tensor_rut_block(&x.data, &y.data, &mut t_d, m, p, n, p, n, n);
        tensor_rut_kernel(&x, &y, &mut t_d);
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_ltl_equivalence(m: usize, p: usize, n: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        let mut t_d = generate_random_vector(m * n);
        filter_lower_trapezoid(&mut x_base);
        x.transpose_inplace(); // stored in transpose
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        // tensor_tlt_block(&x.data, &y.data, &mut t_d, m, p, n, p, n, n);
        // m, n, n b/c X is s tored in it's transposed state
        tensor_tlt_kernel(&x, &y, &mut t_d);
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_ltu_equivalence(m: usize, p: usize, n: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        let mut t_d = generate_random_vector(m * n);
        // filter_lower_trapezoid(&mut x_base);
        filter_upper_trapezoid(&mut x_base);
        x.transpose_inplace(); // stored in transpose
        // println!("x_base {x_base:?}");
        // println!("y {y:?}");
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        // m, n, n b/c X is s tored in it's transposed state
        // tensor_tut_block(&x.data, &y.data, &mut t_d, m, p, n, m, n, n);
        tensor_tut_kernel(&x, &y, &mut t_d);
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
}
