use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stellar::algebra::ndmethods::{parallel_tensor_mult, tensor_mult};
use stellar::algebra::simd::simd_tensor_mult;
use stellar::random::generation::generate_random_matrix;
// use rand::Rng;
// use stellar::structure::ndarray::NdArray;

fn benchmark(c: &mut Criterion) {
    // let small_size = 64;
    // let medium_size = 512;
    // let large_size = 1024;
    let small_size = 64;
    let medium_size = 128;
    let large_size = 256;
    let blocksize = 32; // Example blocksize, modify as needed

    let small_x = generate_random_matrix(small_size, small_size);
    let small_y = generate_random_matrix(small_size, small_size);
    let medium_x = generate_random_matrix(medium_size, medium_size);
    let medium_y = generate_random_matrix(medium_size, medium_size);
    let large_x = generate_random_matrix(large_size, large_size);
    let large_y = generate_random_matrix(large_size, large_size);

    // Small matrix tests
    c.bench_function("small_parallel_tensor_mult", |b| {
        b.iter(|| parallel_tensor_mult(blocksize, black_box(&small_x), black_box(&small_y)))
    });
    c.bench_function("small_tensor_mult", |b| {
        b.iter(|| tensor_mult(blocksize, black_box(&small_x), black_box(&small_y)))
    });
    c.bench_function("small_simd_tensor_mult", |b| {
        b.iter(|| simd_tensor_mult(8, black_box(&small_x), black_box(&small_y)))
    });

    // Medium matrix tests
    c.bench_function("medium_parallel_tensor_mult", |b| {
        b.iter(|| parallel_tensor_mult(blocksize, black_box(&medium_x), black_box(&medium_y)))
    });
    c.bench_function("medium_tensor_mult", |b| {
        b.iter(|| tensor_mult(blocksize, black_box(&medium_x), black_box(&medium_y)))
    });
    c.bench_function("medium_simd_tensor_mult", |b| {
        b.iter(|| simd_tensor_mult(8, black_box(&medium_x), black_box(&medium_y)))
    });

    // Large matrix tests
    c.bench_function("large_parallel_tensor_mult", |b| {
        b.iter(|| parallel_tensor_mult(blocksize, black_box(&large_x), black_box(&large_y)))
    });
    c.bench_function("large_tensor_mult", |b| {
        b.iter(|| tensor_mult(blocksize, black_box(&large_x), black_box(&large_y)))
    });
    c.bench_function("large_simd_tensor_mult", |b| {
        b.iter(|| simd_tensor_mult(8, black_box(&large_x), black_box(&large_y)))
    });
    // Specify the number of iterations for the benchmarks

    //     c.configure_from_args();
    //     c.warm_up_time(std::time::Duration::from_secs(2)); // Optional: set warm-up time
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
