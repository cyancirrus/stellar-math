use crate::sharedvars::{L_MATRIX_DIMS, M_MATRIX_DIMS, S_MATRIX_DIMS};
use criterion::{BenchmarkId, Criterion, black_box};
use stellar::algebra::mmethods::{par_tensor_mult_cache, tensor_mult_cache};
// use stellar::algebra::ndmethods::{tensor_mult, basic_mult};
use stellar::random::generation::generate_random_matrix;

const BLOCK: usize = 8;
pub fn bench_small_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMulImplementation");
    let bsize = BLOCK * BLOCK;

    for &(i, k, j) in S_MATRIX_DIMS.iter() {
        let parameter = format!("{} * {} * {}", i, k, j);
        group.bench_with_input(
            BenchmarkId::new("Small Tensor Mult Cache", &parameter),
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
            BenchmarkId::new("Small Par Tensor Mult Cache", &parameter),
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

pub fn bench_medium_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMulImplementation");
    let bsize = BLOCK * BLOCK;

    for &(i, k, j) in M_MATRIX_DIMS.iter() {
        let parameter = format!("{} * {} * {}", i, k, j);
        group.bench_with_input(
            BenchmarkId::new("Medium Tensor Mult Cache", &parameter),
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
            BenchmarkId::new("Medium Par Tensor Mult Cache", &parameter),
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

pub fn bench_large_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMulImplementation");
    let bsize = BLOCK * BLOCK;

    for &(i, k, j) in L_MATRIX_DIMS.iter() {
        let parameter = format!("{} * {} * {}", i, k, j);
        group.bench_with_input(
            BenchmarkId::new("Large Tensor Mult Cache", &parameter),
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
            BenchmarkId::new("Large Par Tensor Mult Cache", &parameter),
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
