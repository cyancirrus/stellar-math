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
use stellar::arch::SIMD_WIDTH;
use stellar::algebra::ndmethods::basic_mult;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::algebra::mmethods::tensor_kernel;

fn test_kernel_equivalence() {
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
        (16, 8, 16),
    ];
    let block = 8;
    let mut result = vec![f32::NAN; 16 * 16];
    for (i, k, j) in ikj {
        println!("---------------------------------------");
        println!("(i: {i:}, k: {k:}, j: {j:})");
        println!("---------------------------------------");
        test_kernel_equivalence_mkn(block, i, k, j, &mut result);
    }
}
fn test_kernel_equivalence_mkn(block: usize, m: usize, k: usize, n: usize, result: &mut [f32]) {
    let x = generate_random_matrix(m, k);
    let y = generate_random_matrix(k, n);
    let num_threads = rayon::current_num_threads();
    let mut workspace = vec![0f32; block * block * 2 * num_threads];

    let expected = basic_mult(&x, &y);
    tensor_kernel(&x, &y, result, &mut workspace);
    println!("expected {expected:?}");
    println!("result {:?}", &result[..m * n]);
    assert!(approx_vector_eq(&expected.data, &result[..m * n]));
}

fn main() {
    println!("Register width {SIMD_WIDTH:?}");
    test_kernel_equivalence();

}
