use criterion::{Criterion, black_box, criterion_group, criterion_main};
use stellar::kernel::matkerns::{kernel_mult_avx, kernel_mult_scalar};

pub fn benchmark_kernels(c: &mut Criterion) {
    let block = 8;
    let stride = 8;
    let a = vec![1.0f32; 64]; // Use 64 for 8x8 block
    let b = vec![2.0f32; 64];
    let mut c_out = vec![0.0f32; 64];

    let mut group = c.benchmark_group("Matrix Kernel");

    group.bench_function("AVX Kernel", |b_inner| {
        b_inner.iter(|| unsafe {
            kernel_mult_avx(
                black_box(&a),
                black_box(&b),
                black_box(&mut c_out),
                stride,
                0,
            )
        });
    });

    group.bench_function("Scalar Kernel", |b_inner| {
        b_inner.iter(|| {
            kernel_mult_scalar(
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
