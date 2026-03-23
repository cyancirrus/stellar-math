use stellar::decomposition::lq::AutumnDecomp;
use stellar::random::generation::generate_random_matrix;
use criterion::{BenchmarkId, Criterion, black_box};
use ndarray::Array2;
use ndarray_linalg::QR;


pub fn bench_apply_comparisons(c: &mut Criterion) {
    let mut group = c.benchmark_group("Apply_Operations");
    for &n in [16, 32, 64, 128].iter() {
        // 1. Benchmark your custom implementation
        group.bench_with_input(BenchmarkId::new("Autumn_Left_Apply_Q", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let decomp = AutumnDecomp::new(generate_random_matrix(n, n));
                    let target = generate_random_matrix(n, n);
                    let workspace = vec![0.0f32; n];
                    (decomp, target, workspace)
                },
                |(decomp, mut target, mut workspace)| {
                    black_box(decomp.left_apply_q(&mut target, &mut workspace))
                },
            );
        });

        // 2. Benchmark the NDArray equivalent
        group.bench_with_input(BenchmarkId::new("NDArray_Dot_Q", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let raw = generate_random_matrix(n, n);
                    let mat = Array2::from_shape_vec((n, n), raw.data).unwrap();
                    let (q, _) = mat.qr().unwrap();
                    let target = Array2::from_shape_vec((n, n), generate_random_matrix(n, n).data).unwrap();
                    (q, target)
                },
                |(q, target)| {
                    black_box(q.dot(&target))
                },
            );
        });
    }
    group.finish();
}
