use std::hint::black_box;
use std::time::{Duration, Instant};
use stellar::random::generation::generate_random_matrix;
use stellar::solver::randomized_svd::RandomizedSvd;

fn main() {
    let n = 1000;
    let k = 20;
    let iterations = 100;
    let mut total = Duration::ZERO;

    for _ in 0..iterations {
        let x = generate_random_matrix(n, n);
        let x_for_svd = x.clone();

        let start = Instant::now();

        let ksvd = RandomizedSvd::new(k, x_for_svd);
        let tiny = ksvd.approx();
        let big = ksvd.reconstruct();

        total += start.elapsed();
        black_box(tiny);
        black_box(big);
        black_box(x);
    }

    println!("Average Pipeline took: {:?}", total / iterations as u32);
}
