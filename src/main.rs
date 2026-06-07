#![allow(unused)]
// TODO:
// then make the LX, async method
// do the 16 x 16 instruction ie 512 for the tower
// make the toml cfg to get cacheline size etc
// do a small test
// inspect the flamegraph to see if any hanging threads
// ie suspect like communication jam in l1-> l2
//
// value sanity start working on the LX async vision with the queue

// 1. Animate demo        ← most legible to employers
// 2. Blog redesign       ← makes everything else findable
// 3. Triangle kernel     ← 2hrs, unblocks LQ block
// 4. Trait refactor      ← important but least urgent
const MC: usize = 64;
const PC: usize = 256;
const NC: usize = 128;
use stellar::algebra::bmethods::{diff_min, pack};
use stellar::arch::SIMD_WIDTH;
use stellar::kernel::matkerns::{kernel_lt_mult, kernel_ut_mult};
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;

    use stellar::algebra::bmethods_tri::*;
    use stellar::algebra::ndmethods::basic_mult;
    use stellar::equality::approximate::approx_vector_eq;
    use stellar::random::generation::generate_random_matrix;
    use stellar::structure::ndarray::NdArray;
    // #[test]
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
            // upper_equivalence_mkn(i, k, j);
        }
    }
    /// Case 1 / Case 2
    /// -------+---------
    /// * * *  / * * * *
    /// * * *  / 0 * * *
    /// 0 * *  /
    /// 0 0 *  /
    fn filter_upper_triangle(a: &mut NdArray) {
        // i - (m - n);
        let (m, n) = (a.dims[0], a.dims[1]);
        let data = &mut a.data;
        let mut d_sub = if m > n { m - n } else { 0 }; 
        let mut d_plus = 0;
        for i in 0..m {
            for j in 0..n {
                if j + d_sub >= d_plus { break; }
                data[i * n + j] = 0f32;
            }
            d_plus += 1;
        }
    }
    /// * * * * * * * 0 0 0
    /// * * * * * * * * 0 0
    /// * * * * * * * * * 0
    /// * * * * * * * * * *
    fn filter_lower_triangle(a: &mut NdArray) {
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
        filter_lower_triangle(&mut x_base);
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        tensor_lt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    fn upper_equivalence_mkn(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut x_base = x.clone();
        filter_upper_triangle(&mut x_base);
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        tensor_ut_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        let _inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
fn main() {
test_gemm_equivalence();
}
