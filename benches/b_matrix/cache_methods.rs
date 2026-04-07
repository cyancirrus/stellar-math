use crate::sharedvars::{L_MATRIX_DIMS, M_MATRIX_DIMS, S_MATRIX_DIMS};
use criterion::{BenchmarkId, Criterion, Throughput, black_box};
use stellar::algebra::mmethods::{par_tensor_mult_cache, tensor_mult_cache};
use stellar::algebra::ndmethods::{basic_mult, tensor_mult};
use stellar::random::generation::generate_random_matrix;

const BLOCK: usize = 4;
pub fn bench_matmul_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMulImplementation");
    // let active_dims = S_MATRIX_DIMS.iter().chain(M_MATRIX_DIMS.iter()).chain(L_MATRIX_DIMS.iter());
    let active_dims = S_MATRIX_DIMS.iter().chain(M_MATRIX_DIMS.iter());
    let bsize = BLOCK * BLOCK;

    for &(i, k, j) in active_dims {
        let parameter = format!("{}x{}x{}", i, k, j);
        group.throughput(Throughput::Elements((i * k * j) as u64));
        group.sampling_mode(criterion::SamplingMode::Auto);
        group.bench_with_input(
            BenchmarkId::new("naive", &parameter),
            &(i, j, k),
            |b, &(i, j, k)| {
                b.iter_with_setup(
                    || {
                        let x = generate_random_matrix(i, k);
                        let y = generate_random_matrix(k, j);
                        (x, y)
                    },
                    |(x, y)| black_box(tensor_mult(BLOCK, &x, &y)),
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("block", &parameter),
            &(i, j, k),
            |b, &(i, j, k)| {
                b.iter_with_setup(
                    || {
                        let x = generate_random_matrix(i, k);
                        let y = generate_random_matrix(k, j);
                        (x, y)
                    },
                    |(x, y)| black_box(basic_mult(&x, &y)),
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cache", &parameter),
            &(i, j, k),
            |b, &(i, j, k)| {
                b.iter_with_setup(
                    || {
                        let x = generate_random_matrix(i, k);
                        let y = generate_random_matrix(k, j);
                        let work_x = vec![f32::NAN; bsize];
                        let work_y = vec![f32::NAN; bsize];
                        let target = vec![f32::NAN; i * j];
                        (x, y, target, work_x, work_y)
                    },
                    |(x, y, mut target, mut work_x, mut work_y)| {
                        black_box(tensor_mult_cache(
                            &x,
                            &y,
                            &mut target,
                            &mut work_x,
                            &mut work_y,
                            BLOCK,
                        ))
                    },
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("parcache", &parameter),
            &(i, j, k),
            |b, &(i, j, k)| {
                b.iter_with_setup(
                    || {
                        let x = generate_random_matrix(i, k);
                        let y = generate_random_matrix(k, j);
                        let target = vec![f32::NAN; i * j];
                        (x, y, target)
                    },
                    |(x, y, mut target)| {
                        black_box(par_tensor_mult_cache(&x, &y, &mut target, BLOCK))
                    },
                )
            },
        );
    }
    group.finish();
}
