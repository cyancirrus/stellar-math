#![allow(unused)]
use crate::arch::SIMD_WIDTH;
use crate::structure::ndarray::NdArray;
const MC: usize = 16;
const PC: usize = 32;
const NC: usize = 64;

fn test_data() -> Vec<(usize, usize, usize)> {
    vec![
        // (9, 16, 9),
        // (32, 32, 32),
        // (1, 1, 1),
        // (16, 16, 16),
        (8, 9, 8),
        (3, 9, 1),
        (6, 4, 8),
        (9, 16, 8),
        (8, 8, 9),
        (2, 9, 1),
        (2, 2, 1),
        (2, 9, 1),
        (2, 10, 1),
        (1, 9, 1),
        (4, 8, 1),
        (1, 2, 1),
        (1, 1, 1),
        (8, 1, 1),
        (1, 8, 1),
        (1, 1, 8),
        (6, 4, 8),
        (6, 8, 4),
        (8, 4, 6),
        (4, 8, 6),
        (4, 6, 8),
        (8, 6, 4),
        (8, 8, 8),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
        (MC + 1, PC, NC + 1),
        (MC + 1, PC, NC - 1),
        (MC + 1, PC, NC),
        (MC - 1, PC, NC),
        (MC, PC + 1, NC),
        (MC, PC - 1, NC),
        (MC, PC, NC),
        // (256, 256, 256),
        // (256, 1024, 512),
        // (512, 512, 512),
        // (1024, 64, 1024),
    ]
}
fn increment(basis: &mut [f32], data: &[f32], m: usize, n: usize, s_b: usize, s_d: usize) {
    let mut boffset = 0;
    let mut doffset = 0;
    for _ in 0..m {
        for j in 0..n {
            basis[boffset + j] += data[doffset + j];
        }
        boffset += s_b;
        doffset += s_d;
    }
}
/// Case 1 / Case 2
/// -------+---------
/// * * *  / * * * *
/// * * *  / 0 * * *
/// 0 * *  /
/// 0 0 *  /
fn filter_upper_trapezoid(a: &mut NdArray) {
    // i - (m - n);
    let (m, n) = (a.dims[0], a.dims[1]);
    let data = &mut a.data;
    let mut d_sub = if m > n { m - n } else { 0 };
    let mut d_plus = 0;
    for i in 0..m {
        for j in 0..n {
            if j + d_sub >= d_plus {
                break;
            }
            data[i * n + j] = 0f32;
        }
        d_plus += 1;
    }
}
/// * * * * * * * 0 0 0
/// * * * * * * * * 0 0
/// * * * * * * * * * 0
/// * * * * * * * * * *
fn filter_lower_trapezoid(a: &mut NdArray) {
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let d = &mut a.data;
    let t = cols.min(rows);
    let s = rows.saturating_sub(cols);
    // don't remove from last row
    for i in 1..t {
        for j in 0..i {
            d[(rows - i - s) * cols - j - 1] = 0f32;
        }
    }
}
mod test_fma_behavior {
    use super::*;
    use crate::algebra::bmethods::interface::*;
    use crate::algebra::bmethods::contractions::tensor_contraction;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::{generate_random_matrix, generate_random_vector};
    use crate::structure::ndarray::NdArray;
    
    #[test]
    fn test_fma_equivalence() {
        for (i, k, j) in test_data() {
            fma_matmul_equivalence(i, k, j);
            // fma_tmatmul_equivalence(i, k, j);
        }
    }

    fn fma_matmul_equivalence(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut t_d = generate_random_vector(m * n);
        let mut result = vec![0f32; m * n];
        let mut expected = basic_mult(&x, &y);
        increment(&mut expected.data, &t_d, m, n, n, n);
        // tensor_kernel(&x.data, &y.data, &mut result, m, p, n, p, n, n);
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
        increment(&mut expected.data, &t_d, m, p, p, n);
        // let mut result = vec![0f32; m * n];
        // m, n, n b/c X is s tored in it's transposed state
        // tensor_tkernel(&x.data, &y.data, &mut result, m, p, n, m, n, n);
        tensor_tkernel(&x, &y, &mut t_d);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: t_d.clone(),
        };
        // println!("expected {expected:?}");
        // println!("actual {_inspect:?}");
        assert!(
            approx_vector_eq(&expected.data, &t_d[..m * n]),
            "FAILURE WAS ({m:}, {p:}, {n:})"
        );
    }
}
#[cfg(feature = "avx2")]
mod test_kernel_block {
    use super::*;
    use crate::algebra::bmethods::interface::*;
    use crate::algebra::bmethods::contractions::tensor_contraction;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;
    #[test]
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
        let inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
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
#[cfg(feature = "avx2")]
mod test_kernel_trapezoids {
    use super::*;
    use crate::algebra::bmethods::interface::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;
    #[test]
    fn test_gemm_equivalence() {
        for (i, k, j) in test_data() {
            println!("(i: {i:?}, k: {k:?}, j: {j:})");
            lower_equivalence_mkn(i, k, j);
            upper_equivalence_mkn(i, k, j);
            rlower_equivalence_mkn(i, k, j);
            rupper_equivalence_mkn(i, k, j);
            ltl_equivalence_mkn(i, k, j);
            ltu_equivalence_mkn(i, k, j);
        }
    }
    fn lower_equivalence_mkn(m: usize, p: usize, n: usize) {
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
    fn upper_equivalence_mkn(m: usize, p: usize, n: usize) {
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
    fn rlower_equivalence_mkn(m: usize, p: usize, n: usize) {
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
    fn rupper_equivalence_mkn(m: usize, p: usize, n: usize) {
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
    fn ltl_equivalence_mkn(m: usize, p: usize, n: usize) {
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
    fn ltu_equivalence_mkn(m: usize, p: usize, n: usize) {
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
