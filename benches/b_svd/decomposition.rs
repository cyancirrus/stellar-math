use crate::sharedvars::SVD_SIZES;
use criterion::{BenchmarkId, Criterion};
use std::hint::black_box;
use stellar::decomposition::svd::interface::{full_svd_decomposition, svd_decomposition};
use stellar::random::generation::generate_random_matrix;

pub fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVD_Decomposition");

    // Default configuration constants matching your engine
    let max_iters = 40;
    let tolerance = 1e-6;
    let absolute = 1e-4;

    for &n in SVD_SIZES.iter() {
        group.bench_with_input(BenchmarkId::new("Full_Autumn_SVD", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let rows = n;
                    let cols = n;
                    let card = n.min(rows);
                    let stride = cols;

                    let matrix_b = generate_random_matrix(rows, cols);
                    let b = matrix_b.data;
                    let u = vec![0.0f32; rows * rows];
                    let v = vec![0.0f32; cols * cols];
                    let p = vec![0.0f32; rows.max(cols)];
                    let w = vec![0.0f32; rows.max(cols)];

                    (b, u, v, p, w, rows, cols, card, stride)
                },
                |(mut b, mut u, mut v, mut p, mut w, rows, cols, card, stride)| {
                    let _: () = full_svd_decomposition(
                        &mut b, &mut u, &mut v, &mut p, &mut w, rows, cols, card, stride,
                        max_iters, tolerance, absolute,
                    );
                    black_box(())
                },
            );
        });
        group.bench_with_input(BenchmarkId::new("Full_Nalgebra_SVD", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let random_mat = generate_random_matrix(n, n);
                    nalgebra::DMatrix::from_fn(n, n, |i, j| random_mat.data[i * n + j])
                },
                |mat| black_box(nalgebra::SVD::try_new(mat, true, true, 1e-6, 100)),
            );
        });
        group.bench_with_input(BenchmarkId::new("Autumn_SVD", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let rows = n;
                    let cols = n;
                    let card = n.min(rows);
                    let stride = cols;

                    let matrix_b = generate_random_matrix(rows, cols);
                    let b = matrix_b.data;
                    let p = vec![0.0f32; rows.max(cols)];
                    let w = vec![0.0f32; rows.max(cols)];

                    (b, p, w, rows, cols, card, stride)
                },
                |(mut b, mut p, mut w, rows, cols, card, stride)| {
                    let _: () = svd_decomposition(
                        &mut b, &mut p, &mut w, rows, cols, card, stride, max_iters, tolerance,
                        absolute,
                    );
                    black_box(())
                },
            );
        });
        group.bench_with_input(BenchmarkId::new("Nalgebra_SVD", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let random_mat = generate_random_matrix(n, n);
                    nalgebra::DMatrix::from_fn(n, n, |i, j| random_mat.data[i * n + j])
                },
                |mat| black_box(nalgebra::SVD::try_new(mat, false, false, 1e-6, 100)),
            );
        });
    }
    group.finish();
}
