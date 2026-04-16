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
// 4. AVX-512 blocksizes  ← 2hrs, great benchmark result
// 5. Trait refactor      ← important but least urgent
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
use stellar::arch::SIMD_WIDTH;
use stellar::equality::approximate::approx_vector_eq;
use stellar::kernel::matkerns::kernel_mult;

use stellar::algebra::mmethods::{par_tensor_mult_cache, tensor_kernel};
use stellar::algebra::ndmethods::{basic_mult, tensor_mult};
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
// use criterion::{AxisScale, PlotConfiguration};

pub fn tensor_minikern(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    unsafe {
        // will reuse allocation if available
        let x_d = &x.data;
        let y_d = &y.data;
        let (x_rows, x_cols) = (x.dims[0], x.dims[1]);
        let y_cols = y.dims[1];
        let t_d = &mut target[..x_rows * y_cols];
        t_d.fill(0f32);
        debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
        let mut xoffset = 0;
        let mut toffset = 0;
        for i in (0..x_rows).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(x_rows - i);
            let mut yoffset = 0;
            for k in (0..x_cols).step_by(SIMD_WIDTH) {
                let kk_end = SIMD_WIDTH.min(x_cols - k);
                let mut woffset = 0;
                for j in (0..y_cols).step_by(SIMD_WIDTH) {
                    let jj_end = SIMD_WIDTH.min(y_cols - j);
                    kernel_mult(
                        x_d.get_unchecked(xoffset + k ..),
                        y_d.get_unchecked(yoffset + j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        ii_end,
                        kk_end,
                        jj_end,
                        x_cols,
                        y_cols,
                    );
                }
                yoffset += SIMD_WIDTH * y_cols;
            }
            toffset += SIMD_WIDTH * y_cols;
            xoffset += SIMD_WIDTH * x_cols;
        }
    }
}

fn test_minikern_equivalence() {
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
    ];
    let block = 4;
    let mut result = vec![f32::NAN; 8 * 8];
    for (i, k, j) in ikj {
        test_minikern_equivalence_mkn(block, i, k, j, &mut result);
    }
}
fn test_minikern_equivalence_mkn(block: usize, m: usize, k: usize, n: usize, result: &mut [f32]) {
    let x = generate_random_matrix(m, k);
    let y = generate_random_matrix(k, n);
    let mut result = vec![0f32; m * n];

    let expected = basic_mult(&x, &y);
    tensor_minikern(&x, &y, &mut result);
    let inspect = NdArray { dims: vec![m, n], data: result.clone()};
    assert!(approx_vector_eq(&expected.data, &result[..m * n]));
}
fn main() {
    test_minikern_equivalence();
}
