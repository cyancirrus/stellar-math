use crate::sharedvars::LQ_SIZES;
use std::hint::black_box;
// use criterion::{BenchmarkId, Criterion, black_box};
use criterion::{BenchmarkId, Criterion};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::qr::no_pivoting::{factor, solve};
use faer::{Conj, Mat, Par};
use stellar::decomposition::wy::apply::{lhs_apply_q, lhs_apply_qt};
use stellar::decomposition::wy::interface::{easy_solve as autumn_solve, wy_decomposition};
use stellar::random::generation::{generate_random_matrix, generate_random_vector};

/// Compares decomposition cost: Autumn's WY vs faer's QR.
pub fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decomposition");

    for &n in LQ_SIZES.iter() {
        // 1. Your Custom Implementation
        group.bench_with_input(BenchmarkId::new("Autumn_WY", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let a = generate_random_matrix(n, n);
                    let t = vec![0.0f32; n * n];
                    let w = vec![0.0f32; n];
                    (a.data, t, w)
                },
                |(mut a, mut t, mut w)| {
                    wy_decomposition(&mut a, &mut t, &mut w, n, n, n);
                    black_box((a, t))
                },
            );
        });

        // 2. Faer
        group.bench_with_input(BenchmarkId::new("Faer_QR", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let random_a = generate_random_matrix(n, n);
                    let mat_faer = Mat::<f32>::from_fn(n, n, |i, j| random_a.data[i * n + j]);

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

                    (mat_faer, householder_factors, mem)
                },
                |(mut mat_faer, mut householder_factors, mut mem)| {
                    let stack = MemStack::new(&mut mem);
                    factor::qr_in_place(
                        mat_faer.as_mut(),
                        householder_factors.as_mut(),
                        Par::Seq,
                        stack,
                        Default::default(),
                    );
                    black_box(mat_faer)
                },
            );
        });
    }
    group.finish();
}

/// Benchmarks applying Q and Q' via the compact WY representation,
/// each against a freshly-factored matrix so apply cost is isolated
/// from decomposition cost.
pub fn bench_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("WY_Apply");

    for &n in LQ_SIZES.iter() {
        group.bench_with_input(BenchmarkId::new("Apply_Q", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let a = generate_random_matrix(n, n);
                    let mut l_yt = a.data;
                    let mut t = vec![0.0f32; n * n];
                    let mut w = vec![0.0f32; n];
                    wy_decomposition(&mut l_yt, &mut t, &mut w, n, n, n);

                    let x = vec![1.0f32; n * n];
                    let t_buffer = vec![0.0f32; n * n];
                    let s_buffer = vec![0.0f32; n * n];
                    (l_yt, t, x, t_buffer, s_buffer)
                },
                |(l_yt, t, x, mut t_buffer, mut s_buffer)| {
                    lhs_apply_q(&l_yt, &t, &x, &mut t_buffer, &mut s_buffer, n, n, n);
                    black_box(s_buffer)
                },
            );
        });

        group.bench_with_input(BenchmarkId::new("Apply_Qt", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let a = generate_random_matrix(n, n);
                    let mut l_yt = a.data;
                    let mut t = vec![0.0f32; n * n];
                    let mut w = vec![0.0f32; n];
                    wy_decomposition(&mut l_yt, &mut t, &mut w, n, n, n);

                    let x = vec![1.0f32; n * n];
                    let t_buffer = vec![0.0f32; n * n];
                    let s_buffer = vec![0.0f32; n * n];
                    (l_yt, t, x, t_buffer, s_buffer)
                },
                |(l_yt, t, x, mut t_buffer, mut s_buffer)| {
                    lhs_apply_qt(&l_yt, &t, &x, &mut t_buffer, &mut s_buffer, n, n, n);
                    black_box(s_buffer)
                },
            );
        });
    }
    group.finish();
}

/// Compares full solve cost: Autumn's WY-based solve vs faer's QR-based solve.
/// Square case (rows == cols) for a fair apples-to-apples vs faer's solve path.
pub fn bench_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("Solve");

    for &n in LQ_SIZES.iter() {
        let tcols = n; // rhs width; adjust if you want a fixed rhs width instead

        // 1. Autumn's solve
        group.bench_with_input(BenchmarkId::new("Autumn_Solve", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let mut l_yt = generate_random_matrix(n, n).data;
                    let mut tri = vec![0.0f32; n * n];
                    let mut w = vec![0.0f32; n];
                    wy_decomposition(&mut l_yt, &mut tri, &mut w, n, n, n);

                    let y_argument = generate_random_vector(n * tcols);
                    let x_argument = vec![0.0f32; n * tcols];
                    let t_buffer = vec![0.0f32; 8 * n * tcols];
                    let s_buffer = vec![0.0f32; n * tcols];
                    (l_yt, tri, x_argument, y_argument, t_buffer, s_buffer)
                },
                |(l_yt, tri, mut x_argument, y_argument, mut t_buffer, mut s_buffer)| {
                    autumn_solve(
                        &l_yt,
                        &tri,
                        &mut x_argument,
                        &y_argument,
                        &mut t_buffer,
                        &mut s_buffer,
                        n,
                        n,
                        tcols,
                    );
                    black_box(x_argument)
                },
            );
        });

        // 2. Faer's solve, via QR factor then solve_in_place.
        // NOTE: faer's solve API has shifted across versions — verify
        // `solve::solve_in_place_with_conj` (or whatever your pinned version
        // calls it) matches this signature before trusting these numbers.
        group.bench_with_input(BenchmarkId::new("Faer_Solve", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let random_a = generate_random_matrix(n, n);
                    let mut mat_faer = Mat::<f32>::from_fn(n, n, |i, j| random_a.data[i * n + j]);

                    let blocksize = factor::recommended_block_size::<f32>(n, n);
                    let mut householder_factors = Mat::<f32>::zeros(blocksize, n);

                    let req = factor::qr_in_place_scratch::<f32>(
                        n,
                        n,
                        blocksize,
                        Par::Seq,
                        Default::default(),
                    );
                    let mut mem = MemBuffer::new(req);
                    {
                        let stack = MemStack::new(&mut mem);
                        factor::qr_in_place(
                            mat_faer.as_mut(),
                            householder_factors.as_mut(),
                            Par::Seq,
                            stack,
                            Default::default(),
                        );
                    }

                    let rhs_data = generate_random_vector(n * tcols);
                    let rhs = Mat::<f32>::from_fn(n, tcols, |i, j| rhs_data[i * tcols + j]);

                    let solve_req =
                        solve::solve_in_place_scratch::<f32>(n, blocksize, tcols, Par::Seq);
                    let solve_mem = MemBuffer::new(solve_req);

                    (mat_faer, householder_factors, rhs, solve_mem)
                },
                |(mat_faer, householder_factors, mut rhs, mut solve_mem)| {
                    let stack = MemStack::new(&mut solve_mem);
                    solve::solve_in_place_with_conj(
                        mat_faer.as_ref(),
                        householder_factors.as_ref(),
                        mat_faer.as_ref(),
                        Conj::No,
                        rhs.as_mut(),
                        Par::Seq,
                        stack,
                    );
                    black_box(rhs)
                },
            );
        });
    }
    group.finish();
}
