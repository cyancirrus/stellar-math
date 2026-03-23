mod b_lq;
use b_lq::left_apply;
use criterion::{criterion_group, criterion_main};

// criterion_group!(benches_decomp, bench_lq_vs_pure_rust_qr);
criterion_group!(benches_apply, left_apply::bench_apply_comparisons);
criterion_main!(benches_apply);
