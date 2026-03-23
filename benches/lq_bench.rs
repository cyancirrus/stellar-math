use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array2;
use ndarray_linalg::QR;
// use linfa_linalg::qr::QR;

use stellar::decomposition::lq::AutumnDecomp;
use stellar::random::generation::generate_random_matrix;

fn bench_lq_vs_pure_rust_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("Householder_Comparison");

    for n in [16, 32, 64, 128].iter() {
        let size = *n;

        // 1. Your Custom LQ Implementation
        group.bench_with_input(BenchmarkId::new("Autumn_LQ", size), n, |b, &n| {
            b.iter_with_setup(
                || generate_random_matrix(n, n),
                |matrix| black_box(AutumnDecomp::new(matrix)),
            );
        });
        // use ndarray_linalg::QR as LapackQR; // Rename it to avoid conflict with Linfa

        // Inside your bench loop:
        group.bench_with_input(BenchmarkId::new("LAPACK_QR", size), n, |b, &n| {
            b.iter_with_setup(
                || {
                    let raw = generate_random_matrix(n, n);
                    Array2::from_shape_vec((n, n), raw.data).unwrap()
                },
                |mat| {
                    // This calls the underlying C/Fortran LAPACK routine
                    black_box(mat.qr().unwrap())
                },
            );
        });

        // // 2. Pure-Rust QR (from linfa-linalg)
        // group.bench_with_input(BenchmarkId::new("PureRust_QR", size), n, |b, &n| {
        //     b.iter_with_setup(
        //         || {
        //             let raw = generate_random_matrix(n, n);
        //             Array2::from_shape_vec((n, n), raw.data).unwrap()
        //         },
        //         |mat| {
        //             // .qr() is the pure-Rust equivalent logic
        //             black_box(mat.qr().unwrap())
        //         },
        //     );
        // });
    }
    group.finish();
}

fn bench_apply_comparisons(c: &mut Criterion) {
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

criterion_group!(benches_decomp, bench_lq_vs_pure_rust_qr);
criterion_group!(benches_apply, bench_apply_comparisons);
criterion_main!(benches_decomp, benches_apply);
