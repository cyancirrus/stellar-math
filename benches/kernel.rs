mod b_kernel;
mod sharedvars;
use b_kernel::full_block;
use criterion::{criterion_group, criterion_main};
criterion_group!(
    benches_apply,
    full_block::benchmark_kernels,
);
criterion_main!(benches_apply);
