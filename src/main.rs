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
use stellar::algebra::bmethods::tensor_blockkern;
use stellar::algebra::ndmethods::basic_mult;
use stellar::arch::SIMD_WIDTH;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

// check compilation flags
// write a copy fn which actually loads the block

// NOTE: could consider switching to a mask look up table
fn test_blockkern_equivalence() {
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
        println!("(i: {i:?}, k: {k:?}, {j:})");
        test_blockkern_equivalence_mkn(i, k, j);
    }
}
fn test_blockkern_equivalence_mkn(m: usize, k: usize, n: usize) {
    let x = generate_random_matrix(m, k);
    let y = generate_random_matrix(k, n);
    let mut result = vec![0f32; m * n];
    let expected = basic_mult(&x, &y);
    tensor_blockkern(&x.data, &y.data, &mut result, m, k, n);
    let inspect = NdArray {
        dims: vec![m, n],
        data: result.clone(),
    };
    println!("expected {expected:?}");
    println!("actual {inspect:?}");
    assert!(approx_vector_eq(&expected.data, &result[..m * n]));
}

fn main() {
    test_blockkern_equivalence();
}
