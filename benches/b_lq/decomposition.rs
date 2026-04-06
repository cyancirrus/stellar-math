use crate::sharedvars::LQ_SIZES;
use criterion::{BenchmarkId, Criterion, black_box};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::qr::no_pivoting::factor;
use faer::{Mat, Par};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::random::generation::generate_random_matrix;

pub fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Apply_Decomposition");

    for &n in LQ_SIZES.iter() {
        // 1. Your Custom Implementation
        group.bench_with_input(BenchmarkId::new("Autumn_Decomposition", n), &n, |b, &n| {
            b.iter_with_setup(
                || generate_random_matrix(n, n),
                |matrix| black_box(AutumnDecomp::new(matrix)),
            );
        });
        group.bench_with_input(BenchmarkId::new("Faer_Decomposition", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let random_a = generate_random_matrix(n, n);
                    let random_b = generate_random_matrix(n, n);

                    let mat_faer = Mat::<f32>::from_fn(n, n, |i, j| random_a.data[i * n + j]);
                    let target_faer = Mat::<f32>::from_fn(n, n, |i, j| random_b.data[i * n + j]);

                    // Check what's actually in `factor` — it'll be something like:
                    let blocksize = factor::recommended_block_size::<f32>(n, n);
                    let householder_factors = Mat::<f32>::zeros(blocksize, n);

                    let req = factor::qr_in_place_scratch::<f32>(
                        n,
                        n,
                        blocksize,
                        Par::Seq,
                        Default::default(),
                    );
                    let mem = MemBuffer::new(req);

                    (mat_faer, householder_factors, target_faer, mem)
                },
                |(mut mat_faer, mut householder_factors, target_faer, mut mem)| {
                    let stack = MemStack::new(&mut mem);

                    factor::qr_in_place(
                        mat_faer.as_mut(),
                        householder_factors.as_mut(),
                        Par::Seq,
                        stack,
                        Default::default(),
                    );
                    black_box(target_faer)
                },
            );
        });
    }
    group.finish();
}
