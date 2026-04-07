mod b_matrix;
mod sharedvars;
use b_matrix::cache_methods;
use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches_performance,
    cache_methods::bench_small_matmul,
    cache_methods::bench_medium_matmul,
    // cache_methods::bench_large_matmul,
);
criterion_main!(benches_performance);
