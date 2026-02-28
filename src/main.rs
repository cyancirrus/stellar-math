use std::hint::black_box;
use std::time::{Duration, Instant};
// use stellar::algebra::ndmethods::matrix_mult;
use stellar::random::generation::generate_random_matrix;
use stellar::solver::randomized_svd::RandomizedSvd;
// use stellar::structure::ndarray::NdArray;

fn main() {
    let n = 20;
    let iterations = 100;
    let mut total = Duration::ZERO;

    for _ in 0..iterations {
        // 1. Setup phase (not timed)
        let x = generate_random_matrix(n, n);
        // If your SVD takes ownership, clone it here:
        let x_for_svd = x.clone();

        // 2. Timing start
        let start = Instant::now();

        // 3. Work phase
        // Pass the pre-cloned x here
        let ksvd = RandomizedSvd::new(20, x_for_svd);
        let tiny = ksvd.approx();
        let big = ksvd.reconstruct();

        // 4. Timing end
        total += start.elapsed();

        // Prevent compiler from optimizing out the results
        black_box(tiny);
        black_box(big);
        black_box(x);
    }

    println!("Average Pipeline took: {:?}", total / iterations as u32);
}
