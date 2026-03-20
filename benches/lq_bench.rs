use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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
        use ndarray_linalg::QR as LapackQR; // Rename it to avoid conflict with Linfa

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

criterion_group!(benches, bench_lq_vs_pure_rust_qr);
criterion_main!(benches);
