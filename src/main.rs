#![allow(unused)]
// use stellar::algebra::ndmethods::{basic_mult, create_identity_matrix, tensor_mult};
// use stellar::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
// use stellar::decomposition::lq::AutumnDecomp;
// use stellar::equality::approximate::approx_vector_eq;
// use stellar::random::generation::generate_random_matrix;
// use rayon::prelude::*;
// use rayon::slice::ParallelSlice;
// use stellar::kernel::matkerns::kernel_mult;
// use stellar::structure::ndarray::NdArray;

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


fn thing(
    x: &[f32],
    y: &[f32],
    target: &mut [f32],
    workspace: &mut [f32],
    m:usize, k:usize, n:usize,
    sx:usize, sy:usize,
    block:usize,
    xt:bool,
    yt:bool,
){

}


fn main() {
}
