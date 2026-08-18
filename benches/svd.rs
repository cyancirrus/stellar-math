mod b_svd;
mod sharedvars;
use b_svd::decomposition;
use criterion::{criterion_group, criterion_main};

criterion_group!(benches_apply, decomposition::bench_decomposition,);
criterion_main!(benches_apply);
