#[cfg(test)]
#[cfg(feature = "avx2")]
mod test_stride_behavior {
    use crate::algebra::bmethods::diagonals::*;
    use crate::algebra::bmethods::interface::*;
    use crate::algebra::bmethods::tests::helpers::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::{generate_random_matrix, generate_random_vector};

    #[test]
    fn test_sfma_equivalence() {
        let dsx = 4;
        let dsy = 8;
        for (i, k, j) in test_data() {
            let sx = dsx + k;
            let sy = dsy + j;
            let st = j;
            sfma_matmul_equivalence(i, k, j, sx, sy, st);
            sfma_lower_equivalence(i, k, j, sx, sy, st);
            sfma_upper_equivalence(i, k, j, sx, sy, st);
            sfma_rlower_equivalence(i, k, j, sx, sy, st);
            sfma_rupper_equivalence(i, k, j, sx, sy, st);
        }
        for (i, k, j) in test_data() {
            let sx = dsx + i;
            let sy = dsy + j;
            let st = j;
            sfma_tmatmul_equivalence(i, k, j, sx, sy, st);
            sfma_ltl_equivalence(i, k, j, sx, sy, st);
            sfma_ltu_equivalence(i, k, j, sx, sy, st);
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
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_tmatmul_equivalence(m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let x_base = x.clone();
        let mut t_d = generate_random_vector(m * n);
        x.transpose_inplace();
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        let mut x_stride = vec![0f32; p * s_x];
        let mut y_stride = vec![0f32; p * s_y];
        pack_stride(&mut x_stride, &x.data, p, m, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        stride_tkernel(&x_stride, &y_stride, &mut t_d, m, p, n, s_x, s_y, s_t);
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
        let mut t_d = generate_random_vector(m * n);
        let mut x_base = x.clone();
        filter_lower_trapezoid(&mut x_base);
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);

        let (d_add, d_sub) = diagonal_lt(m, p, n);
        pack_stride(&mut x_stride, &x.data, m, p, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        stride_lt_kernel(
            &x_stride, &y_stride, &mut t_d, d_add, d_sub, m, p, n, s_x, s_y, s_t,
        );
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_upper_equivalence(m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut t_d = generate_random_vector(m * n);
        let mut x_base = x.clone();
        filter_upper_trapezoid(&mut x_base);
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        let mut x_stride = vec![0f32; m * s_x];
        let mut y_stride = vec![0f32; p * s_y];
        let (d_add, d_sub) = diagonal_ut(m, p, n);
        pack_stride(&mut x_stride, &x.data, m, p, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        stride_ut_kernel(
            &x_stride, &y_stride, &mut t_d, d_add, d_sub, m, p, n, s_x, s_y, s_t,
        );
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_rlower_equivalence(m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut t_d = generate_random_vector(m * n);
        let mut y_base = y.clone();
        filter_lower_trapezoid(&mut y_base);
        let mut expected = basic_mult(&x, &y_base);
        increment(&mut expected.data, &t_d, m, n, n, n);
        let (d_add, d_sub) = diagonal_rlt(m, p, n);
        let mut x_stride = vec![0f32; m * s_x];
        let mut y_stride = vec![0f32; p * s_y];
        pack_stride(&mut x_stride, &x.data, m, p, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        stride_rlt_kernel(
            &x_stride, &y_stride, &mut t_d, d_add, d_sub, m, p, n, s_x, s_y, s_t,
        );
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_rupper_equivalence(m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut t_d = generate_random_vector(m * n);
        let mut y_base = y.clone();
        filter_upper_trapezoid(&mut y_base);
        let mut expected = basic_mult(&x, &y_base);
        increment(&mut expected.data, &t_d, m, n, n, n);
        let mut x_stride = vec![0f32; m * s_x];
        let mut y_stride = vec![0f32; p * s_y];
        let (d_add, d_sub) = diagonal_rut(m, p, n);
        pack_stride(&mut x_stride, &x.data, m, p, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        stride_rut_kernel(
            &x_stride, &y_stride, &mut t_d, d_add, d_sub, m, p, n, s_x, s_y, s_t,
        );
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_ltl_equivalence(m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        let mut t_d = generate_random_vector(m * n);
        filter_lower_trapezoid(&mut x_base);
        x.transpose_inplace(); // stored in transpose
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        let mut x_stride = vec![0f32; p * s_x];
        let mut y_stride = vec![0f32; p * s_y];
        let (d_add, d_sub) = diagonal_tlt(m, p, n);
        pack_stride(&mut x_stride, &x.data, p, m, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        stride_tlt_kernel(
            &x_stride, &y_stride, &mut t_d, d_add, d_sub, m, p, n, s_x, s_y, s_t,
        );
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
    fn sfma_ltu_equivalence(m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
        let y = generate_random_matrix(p, n);
        let mut x = generate_random_matrix(m, p);
        let mut x_base = x.clone();
        let mut t_d = generate_random_vector(m * n);
        // filter_lower_trapezoid(&mut x_base);
        filter_upper_trapezoid(&mut x_base);
        x.transpose_inplace(); // stored in transpose
        let mut expected = basic_mult(&x_base, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        let mut x_stride = vec![0f32; p * s_x];
        let mut y_stride = vec![0f32; p * s_y];
        let (d_add, d_sub) = diagonal_tut(m, p, n);
        pack_stride(&mut x_stride, &x.data, p, m, s_x);
        pack_stride(&mut y_stride, &y.data, p, n, s_y);
        stride_tut_kernel(
            &x_stride, &y_stride, &mut t_d, d_add, d_sub, m, p, n, s_x, s_y, s_t,
        );
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
}
