use crate::sizes::LQ_SIZES;
use criterion::{BenchmarkId, Criterion, black_box};
use faer::linalg::matmul::triangular::{self, BlockStructure};
use faer::{Accum, Mat, Par};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::random::generation::generate_random_matrix;

pub fn bench_apply_left_t(c: &mut Criterion) {
    let mut group = c.benchmark_group("Apply_Operations");

    for &n in LQ_SIZES.iter() {
        // 1. Your Custom Implementation
        group.bench_with_input(BenchmarkId::new("Autumn_Left_Apply_T", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let decomp = AutumnDecomp::new(generate_random_matrix(n, n));
                    let target = generate_random_matrix(n, n);
                    (decomp, target, n)
                },
                |(decomp, mut target, n)| {
                    let mut workspace = vec![0.0f32; n];
                    black_box(decomp.left_apply_l(&mut target, &mut workspace))
                },
            );
        });

        group.bench_with_input(BenchmarkId::new("Faer_TriMul_Left_R", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let random_a = generate_random_matrix(n, n);
                    let random_b = generate_random_matrix(n, n);

                    // upper triangular R — zero out lower triangle
                    let r = Mat::<f32>::from_fn(n, n, |i, j| {
                        if i <= j {
                            random_a.data[i * n + j]
                        } else {
                            0.0
                        }
                    });
                    let target = Mat::<f32>::from_fn(n, n, |i, j| random_b.data[i * n + j]);
                    let dst = Mat::<f32>::zeros(n, n);

                    (r, target, dst)
                },
                |(r, target, mut dst)| {
                    triangular::matmul(
                        dst.as_mut(),
                        BlockStructure::Rectangular,
                        Accum::Replace, // beta — Replace means dst = alpha * lhs * rhs
                        r.as_ref(),
                        BlockStructure::TriangularUpper,
                        target.as_ref(),
                        BlockStructure::Rectangular,
                        1.0f32, // alpha
                        Par::Seq,
                    );
                    black_box(dst)
                },
            );
        });
    }
    group.finish();
}
