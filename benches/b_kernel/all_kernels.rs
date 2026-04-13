use criterion::{Criterion, black_box, criterion_group, criterion_main};
#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use stellar::kernel::avx2;
use stellar::kernel::default;

pub fn benchmark_kernels(c: &mut Criterion) {
    let block = 8;
    let stride = 8;
    let a = vec![1.0f32; 64]; // Use 64 for 8x8 block
    let b = vec![2.0f32; 64];
    let mut c_out = vec![0.0f32; 64];

    let mut group = c.benchmark_group("Matrix Kernel");
    #[cfg(feature = "avx2")]
    group.bench_function("AVX2 Kernel", |b_inner| {
        b_inner.iter(|| unsafe {
            avx2::kernel_mult_simd(
                black_box(&a),
                black_box(&b),
                black_box(&mut c_out),
                block,
                stride,
                0,
            )
        });
    });

    group.bench_function("Default Kernel", |b_inner| {
        b_inner.iter(|| unsafe {
            default::kernel_mult_simd(
                black_box(&a),
                black_box(&b),
                black_box(&mut c_out),
                block,
                stride,
                0,
            )
        });
    });

    group.bench_function("Scalar Kernel", |b_inner| {
        b_inner.iter(|| {
            default::kernel_mult_scalar(
                black_box(&a),
                black_box(&b),
                black_box(&mut c_out),
                block,
                block,
                block,
                stride,
                0,
            )
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_kernels);
criterion_main!(benches);
