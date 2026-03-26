use crate::sizes::LQ_SIZES;
use criterion::{BenchmarkId, Criterion, black_box};
use faer::Mat;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder;
use faer::linalg::qr::no_pivoting::factor;
use faer::{Conj, Par};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::random::generation::generate_random_matrix;

pub fn bench_apply_right_q(c: &mut Criterion) {
    let mut group = c.benchmark_group("Apply_Operations");

    for &n in LQ_SIZES.iter() {
        // 1. Your Custom Implementation
        group.bench_with_input(BenchmarkId::new("Autumn_Right_Apply_Q", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let decomp = AutumnDecomp::new(generate_random_matrix(n, n));
                    let target = generate_random_matrix(n, n);
                    let workspace = vec![f32::NAN; n];
                    (decomp, target, workspace)
                },
                |(decomp, mut target, mut workspace)| {
                    black_box(decomp.right_apply_q(&mut target, &mut workspace))
                },
            );
        });

        group.bench_with_input(BenchmarkId::new("Faer_Right_Apply_Q", n), &n, |b, &n| {
    b.iter_with_setup(
        || {
            let random_a = generate_random_matrix(n, n);
            let random_b = generate_random_matrix(n, n);

            let mut mat_faer = Mat::<f32>::from_fn(n, n, |i, j| random_a.data[i * n + j]);
            let target_faer = Mat::<f32>::from_fn(n, n, |i, j| random_b.data[i * n + j]);

            // Check what's actually in `factor` — it'll be something like:
            let blocksize = factor::recommended_block_size::<f32>(n, n);
            let mut householder_factors = Mat::<f32>::zeros(blocksize, n);

            let req = factor::qr_in_place_scratch::<f32>(n, n, blocksize, Par::Seq, Default::default());
            let mut mem = MemBuffer::new(req);
            let stack = MemStack::new(&mut mem);

            factor::qr_in_place(
                mat_faer.as_mut(),
                householder_factors.as_mut(),
                Par::Seq,
                stack,
                Default::default(),
            );
            // Q application lives in householder module — look for apply_block_householder_sequence_*
            let req = householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f32>(
                n, householder_factors.nrows(), n,
            );
            let mem = MemBuffer::new(req);

            (mat_faer, householder_factors, target_faer, mem)
        },
        |(mat_faer, householder_factors, mut target_faer, mut mem)| {
            let stack = MemStack::new(&mut mem);

            householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
                mat_faer.as_ref(),
                householder_factors.as_ref(),
                Conj::No,
                target_faer.as_mut(),
                Par::Seq,
                stack,
            );
            black_box(target_faer)
        },
    );
});
    }
    group.finish();
}

pub fn bench_apply_right_qt(c: &mut Criterion) {
    let mut group = c.benchmark_group("Apply_Operations");

    for &n in LQ_SIZES.iter() {
        // 1. Your Custom Implementation
        group.bench_with_input(BenchmarkId::new("Autumn_Right_Apply_Qt", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let decomp = AutumnDecomp::new(generate_random_matrix(n, n));
                    let target = generate_random_matrix(n, n);
                    (decomp, target)
                },
                |(decomp, mut target)| black_box(decomp.right_apply_qt(&mut target)),
            );
        });

        group.bench_with_input(BenchmarkId::new("Faer_Right_Apply_Qt", n), &n, |b, &n| {
    b.iter_with_setup(
        || {
            let random_a = generate_random_matrix(n, n);
            let random_b = generate_random_matrix(n, n);

            let mut mat_faer = Mat::<f32>::from_fn(n, n, |i, j| random_a.data[i * n + j]);
            let target_faer = Mat::<f32>::from_fn(n, n, |i, j| random_b.data[i * n + j]);

            // Check what's actually in `factor` — it'll be something like:
            let blocksize = factor::recommended_block_size::<f32>(n, n);
            let mut householder_factors = Mat::<f32>::zeros(blocksize, n);

            let req = factor::qr_in_place_scratch::<f32>(n, n, blocksize, Par::Seq, Default::default());
            let mut mem = MemBuffer::new(req);
            let stack = MemStack::new(&mut mem);

            factor::qr_in_place(
                mat_faer.as_mut(),
                householder_factors.as_mut(),
                Par::Seq,
                stack,
                Default::default(),
            );
            // Q application lives in householder module — look for apply_block_householder_sequence_*
            let req = householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f32>(
                n, householder_factors.nrows(), n,
            );
            let mem = MemBuffer::new(req);

            (mat_faer, householder_factors, target_faer, mem)
        },
        |(mat_faer, householder_factors, mut target_faer, mut mem)| {
            let stack = MemStack::new(&mut mem);

            householder::apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj(
                mat_faer.as_ref(),
                householder_factors.as_ref(),
                Conj::No,
                target_faer.as_mut(),
                Par::Seq,
                stack,
            );
            black_box(target_faer)
        },
    );
});
    }
    group.finish();
}
