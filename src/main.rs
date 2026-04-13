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

use stellar::algebra::mmethods::{
    par_tensor_mult_cache, tensor_kernel, tensor_kernel_new, tensor_mult_cache,
};
use stellar::algebra::ndmethods::{basic_mult, tensor_mult};
use stellar::arch::SIMD_WIDTH;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
// use criterion::{AxisScale, PlotConfiguration};

const BLOCK_ITER: usize = 64;
const BLOCK_CACHE: usize = 64;
const BLOCK_CACHE_PAR: usize = 8;

const NUM_THREADS: usize = 4;

fn quick_test(i: usize, k: usize, j: usize) {
    let x = generate_random_matrix(i, k);
    let y = generate_random_matrix(k, j);
    let mut target = vec![0.0f32; i * j];
    // let mut workspace = vec![0.0f32; 2 * (j + 1)  * SIMD_WIDTH * SIMD_WIDTH];
    // let mut workspace = vec![0.0f32; (i + SIMD_WIDTH) * SIMD_WIDTH];
    // let mut workspace = vec![0.0f32; (i + SIMD_WIDTH) * SIMD_WIDTH];

    // Run your kernel
    let dims = vec![i, j];
    // tensor_kernel(&x, &y, &mut target, &mut workspace);
    tensor_kernel_new(&x, &y, &mut target);
    let actual = NdArray {
        dims: dims,
        data: target,
    };

    let expected = basic_mult(&x, &y);

    println!("actual {actual:?}");
    println!("expected {expected:?}");
}

fn main() {
    // inspect(64, 128, 64);
    quick_test(31, 128, 18);
    // quick_test(20, 36, 18);
    // quick_test(17, 2, 4);
    // quick_test(32, 36, 32);
}
