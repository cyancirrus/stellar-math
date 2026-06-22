#![allow(unused)]
use crate::arch::SIMD_WIDTH;
use crate::kernel::matkerns::{kernel_mult, kernel_tmult};
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
#[rustfmt::skip]
use crate::algebra::bmethods::contractions::{
    tensor_contraction,
    tensor_tcontraction,
    tensor_lt_contraction,
    tensor_ut_contraction,
    tensor_rlt_contraction,
    tensor_rut_contraction,
    tensor_tlt_contraction,
    tensor_tut_contraction,
};
#[rustfmt::skip]
use crate::algebra::bmethods::blocks:: {
    tensor_block,
    tensor_tblock,
    tensor_lt_block,
    tensor_ut_block,
    tensor_rlt_block,
    tensor_rut_block,
    tensor_tlt_block,
    tensor_tut_block,
};

const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;

///  tensor_kernel
///  - accumulates the multiplication into the target matrix
///  - t += x * y
///  tensor_kernel_into
///  # not accumulated
///   - returns t = x * y

#[inline(always)]
pub fn tensor_kernel_into(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    target.fill(0f32);
    tensor_kernel(x, y, target);
}

#[inline(always)]
pub fn tensor_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_contraction(&x.data, &y.data, target, m, p, n, p, n, n);
    } else {
        tensor_block(&x.data, &y.data, target, m, p, n, p, n, n);
    }
}
#[inline(always)]
pub fn tensor_tkernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[0], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[1], y.dims[0], y.dims[1]);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_tcontraction(&x.data, &y.data, target, m, p, n, m, n, n);
    } else {
        tensor_tblock(&x.data, &y.data, target, m, p, n, m, n, n);
    }
}
#[inline(always)]
pub fn tensor_lt_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (p - p.min(m) + 1, 0 );
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_lt_contraction(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
    } else {
        tensor_lt_block(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
    }
}
#[inline(always)]
pub fn tensor_ut_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (0, m.saturating_sub(p));
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_ut_contraction(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
    } else {
        tensor_ut_block(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
    }
}
#[inline(always)]
pub fn tensor_rlt_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (n - n.min(p), 0 );
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_rlt_contraction(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
    } else {
        tensor_rlt_block(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
    }
}
#[inline(always)]
pub fn tensor_rut_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (p - p.min(n) + 1, 0 );
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_rut_contraction(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
    } else {
        tensor_rut_block(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
    }
}
#[inline(always)]
pub fn tensor_tlt_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[0], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[1], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (p - p.min(m) + 1, 0 );
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_tlt_contraction(&x.data, &y.data, target, d_add, d_sub, m, p, n, m, n, n);
    } else {
        tensor_tlt_block(&x.data, &y.data, target, d_add, d_sub, m, p, n, m, n, n);
    }
}
#[inline(always)]
pub fn tensor_tut_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[0], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[1], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (0, m.saturating_sub(p));
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_tut_contraction(&x.data, &y.data, target, d_add, d_sub, m, p, n, m, n, n);
    } else {
        tensor_tut_block(&x.data, &y.data, target, d_add, d_sub, m, p, n, m, n, n);
    }
}

#[cfg(test)]
mod test_kernel_block {
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::arch::SIMD_WIDTH;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;
    
    const MC: usize = 16;
    const PC: usize = 32;
    const NC: usize = 64;

    #[test]
    fn test_outkern_equivalence() {
        let ikj = [
            (1, 1, 1),
            (8, 1, 1),
            (1, 8, 1),
            (1, 1, 8),
            (6, 4, 8),
            (6, 8, 4),
            (4, 6, 8),
            (4, 8, 6),
            (8, 4, 6),
            (8, 6, 4),
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
        ];
        for (i, k, j) in ikj {
            test_outkern_equivalence_mkn(i, k, j);
        }
    }
    fn test_outkern_equivalence_mkn(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut result = vec![0f32; m * n];
        let expected = basic_mult(&x, &y);
        tensor_contraction(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        // let inspect = NdArray {
        //     dims: vec![m, n],
        //     data: result.clone(),
        // };
        // println!("expected {expected:?}");
        // println!("actual {inspect:?}");
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    #[test]
    fn test_gemm_equivalence() {
        let ikj = [
            // (256, 256, 256),
            (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
            (1, 1, 1),
            (8, 1, 1),
            (1, 8, 1),
            (1, 1, 8),
            (6, 4, 8),
            (6, 8, 4),
            (4, 6, 8),
            (4, 8, 6),
            (8, 4, 6),
            (8, 6, 4),
            (8, 8, 8),
            (16, 16, 16),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
            (256, 1024, 512),
            (512, 512, 512),
            (1024, 64, 1024),
        ];
        for (i, k, j) in ikj {
            println!("(i: {i:?}, k: {k:?}, j: {j:})");
            matmul_equivalence(i, k, j);
            tmatmul_equivalence(i, k, j);
        }
    }
    fn matmul_equivalence(m: usize, k: usize, n: usize) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);
        let mut result = vec![0f32; m * n];
        let expected = basic_mult(&x, &y);
        // tensor_kernel(&x.data, &y.data, &mut result, m, k, n, k, n, n);
        tensor_kernel(&x, &y, &mut result);
        let inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        println!("expected {expected:?}");
        println!("actual {inspect:?}");
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
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
#[cfg(test)]
#[cfg(feature = "avx2")]
mod test_kernel_trapezoids{
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;
    const MC: usize = 16;
    const PC: usize = 32;
    const NC: usize = 64;
    #[test]
    fn test_gemm_equivalence() {
        let ikj = [
            (9, 16, 9),
            (32, 32, 32),
            (1, 1, 1),
            (16, 16, 16),
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
            (256, 256, 256),
            (256, 1024, 512),
            (512, 512, 512),
            (1024, 64, 1024),
        ];
        for (i, k, j) in ikj {
            println!("(i: {i:?}, k: {k:?}, j: {j:})");
            lower_equivalence_mkn(i, k, j);
            upper_equivalence_mkn(i, k, j);
            rlower_equivalence_mkn(i, k, j);
            rupper_equivalence_mkn(i, k, j);
            ltl_equivalence_mkn(i, k, j);
            ltu_equivalence_mkn(i, k, j);
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
