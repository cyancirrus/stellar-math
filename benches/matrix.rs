mod b_matrix;
mod sharedvars;
use b_matrix::matmul;
use criterion::{criterion_group, criterion_main};

criterion_group!(benches_performance, matmul::bench_matmul_scaling,);
criterion_main!(benches_performance);
