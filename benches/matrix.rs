mod b_matrix;
mod sharedvars;
use b_matrix::cache_methods;
use criterion::{criterion_group, criterion_main};

criterion_group!(benches_performance, cache_methods::bench_matmul_scaling,);
criterion_main!(benches_performance);
