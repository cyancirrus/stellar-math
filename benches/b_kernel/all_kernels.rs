use criterion::Throughput;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use stellar::arch::SIMD_WIDTH;
#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use stellar::kernel::avx2;
use stellar::kernel::default;
pub fn benchmark_kernels(c: &mut Criterion) {
    let block = SIMD_WIDTH;
    let stride = SIMD_WIDTH;
    let a = vec![1.0f32; SIMD_WIDTH * SIMD_WIDTH]; // Use 64 for 8x8 block
    let b = vec![2.0f32; SIMD_WIDTH * SIMD_WIDTH];
    let mut c_out = vec![0.0f32; SIMD_WIDTH * SIMD_WIDTH];

    let mut group = c.benchmark_group("Matrix Kernel");
    group.sampling_mode(criterion::SamplingMode::Auto);
    group.throughput(Throughput::Elements(
        (2 * SIMD_WIDTH * SIMD_WIDTH * SIMD_WIDTH) as u64,
    ));
    #[cfg(feature = "avx2")]
    group.bench_function("AVX2 Kernel", |b_inner| {
        b_inner.iter(|| unsafe {
            avx2::kernel_mult_simd(
                black_box(a.as_ptr()),
                black_box(b.as_ptr()),
                black_box(c_out.as_mut_ptr()),
                block,
                stride,
                stride,
                stride,
            )
        });
    });
    #[cfg(feature = "avx2")]
    group.bench_function("AVX2 Outer Kernel", |b_inner| {
        b_inner.iter(|| {
            unsafe {
            avx2::kernel_imult_simd(
                black_box(a.as_ptr()),
                black_box(b.as_ptr()),
                black_box(c_out.as_mut_ptr()),
                stride,
                stride,
                stride,
            )}
        });
    });
    group.bench_function("Scalar Kernel", |b_inner| {
        b_inner.iter(|| {
            default::kernel_mult_scalar(
                black_box(a.as_ptr()),
                black_box(b.as_ptr()),
                black_box(c_out.as_mut_ptr()),
                block,
                block,
                block,
                stride,
                stride,
                stride,
            )
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_kernels);
criterion_main!(benches);
