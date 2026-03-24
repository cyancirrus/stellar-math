use stellar::decomposition::lq::AutumnDecomp;
use stellar::random::generation::generate_random_matrix;
use criterion::{BenchmarkId, Criterion, black_box};
use faer::Mat;


pub fn bench_apply_comparisons(c: &mut Criterion) {
    let mut group = c.benchmark_group("Apply_Operations");
    
    for &n in [16, 32, 64, 128].iter() {
        // 1. Your Custom Implementation
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



group.bench_with_input(BenchmarkId::new("Faer_Left_Apply_Q", n), &n, |b, &n| {
    b.iter_with_setup(
        || {
            let random_a = generate_random_matrix(n, n);
            let random_b = generate_random_matrix(n, n);

            let mat_faer = Mat::<f32>::from_fn(n, n, |i, j| random_a.data[i * n + j]);
            let target_faer = Mat::<f32>::from_fn(n, n, |i, j| random_b.data[i * n + j]);

            // qr() returns a high-level Qr type
            let qr = mat_faer.qr();

            (qr, target_faer)
        },
        |(qr, target_faer)| {
            // The actual Q matrix — this is the explicit form
            // compute_thin_q() or just multiplying by the stored factors
            let q = qr.compute_thin_Q();
            // Q * target: left-apply Q to target
            black_box(q * target_faer)
        },
    );
});
}
    group.finish();
}
