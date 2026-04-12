mod b_kernel;
mod sharedvars;
use b_kernel::all_kernels;
use criterion::{criterion_group, criterion_main};

criterion_group!(benches_apply, all_kernels::benchmark_kernels,);
criterion_main!(benches_apply);
