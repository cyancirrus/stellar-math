use crate::sharedvars::{L_MATRIX_DIMS, M_MATRIX_DIMS, S_MATRIX_DIMS};
use criterion::{BenchmarkId, Criterion, Throughput, black_box};
use faer::linalg::matmul::matmul;
use faer::prelude::*;
use faer::prelude::*;
use ndarray::Array2;
use stellar::algebra::mmethods::{par_tensor_mult_cache, tensor_mult_cache};
use stellar::algebra::ndmethods::{basic_mult, tensor_mult};
use stellar::random::generation::generate_random_matrix;
// use criterion::{AxisScale, PlotConfiguration};

const BLOCK_ITER: usize = 64;
const BLOCK_CACHE: usize = 64;
const BLOCK_CACHE_PAR: usize = 8;

pub fn bench_matmul_scaling(c: &mut Criterion) {
    let mut run_bench = |group_name: &str, dims: &[(usize, usize, usize)]| {
        let mut group = c.benchmark_group(group_name);
        // NOTE: would need an external library
        // let plot_config = PlotConfiguration::default()
        // .summary_scale(AxisScale::Logarithmic);
        // group.plot_config(plot_config);
        group.sampling_mode(criterion::SamplingMode::Auto);
        for &(i, k, j) in dims {
            let parameter = format!("{}x{}x{}", i, k, j);
            group.throughput(Throughput::Elements((2 * i * k * j) as u64));
            group.bench_with_input(
                BenchmarkId::new("faer", &parameter),
                &(i, j, k),
                |b, &(m, n, k)| {
                    b.iter_with_setup(
                        || {
                            let data_x = generate_random_matrix(m, k);
                            let data_y = generate_random_matrix(k, n);
                            let x = faer::Mat::from_fn(m, k, |r, c| data_x.data[r * k + c]);
                            let y = faer::Mat::from_fn(k, n, |r, c| data_y.data[r * n + c]);
                            let target = faer::Mat::<f32>::zeros(m, n);
                            (x, y, target)
                        },
                        |(x, y, mut target)| {
                            matmul(
                                target.as_mut(),
                                faer::Accum::Replace,
                                x.as_ref(),
                                y.as_ref(),
                                1.0f32,
                                faer::Par::Rayon(std::num::NonZero::new(8).unwrap()),
                            );
                            black_box(target)
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
                            let num_threads = rayon::current_num_threads();
                            let workspace =
                                vec![0f32; BLOCK_CACHE_PAR * BLOCK_CACHE_PAR * 2 * num_threads];
                            let x = generate_random_matrix(i, k);
                            let y = generate_random_matrix(k, j);
                            let target = vec![f32::NAN; i * j];
                            (x, y, target, workspace)
                        },
                        |(x, y, mut target, mut workspace)| {
                            black_box(par_tensor_mult_cache(
                                &x,
                                &y,
                                &mut target,
                                &mut workspace,
                                BLOCK_CACHE_PAR,
                            ))
                        },
                    )
                },
            );
            // group.bench_with_input(
            //     BenchmarkId::new("ndarray", &parameter),
            //     &(i, j, k),
            //     |b, &(i, j, k)| {
            //         b.iter_with_setup(
            //             || {
            //                 let data_x = generate_random_matrix(i, k);
            //                 let data_y = generate_random_matrix(k, j);
            //                 let x = Array2::from_shape_vec((i, k), data_x.data).unwrap();
            //                 let y = Array2::from_shape_vec((k, j), data_y.data).unwrap();
            //                 (x, y)
            //             },
            //             |(x, y)| {
            //                 black_box(x.dot(&y))
            //             },
            //         );
            //     },
            // );
            // group.bench_with_input(
            //     BenchmarkId::new("naive", &parameter),
            //     &(i, j, k),
            //     |b, &(i, j, k)| {
            //         b.iter_with_setup(
            //             || {
            //                 let x = generate_random_matrix(i, k);
            //                 let y = generate_random_matrix(k, j);
            //                 (x, y)
            //             },
            //             |(x, y)| black_box(basic_mult(&x, &y)),
            //         );
            //     },
            // );
            // group.bench_with_input(
            //     BenchmarkId::new("block", &parameter),
            //     &(i, j, k),
            //     |b, &(i, j, k)| {
            //         b.iter_with_setup(
            //             || {
            //                 let x = generate_random_matrix(i, k);
            //                 let y = generate_random_matrix(k, j);
            //                 (x, y)
            //             },
            //             |(x, y)| black_box(tensor_mult(BLOCK_ITER, &x, &y)),
            //         );
            //     },
            // );
            // group.bench_with_input(
            //     BenchmarkId::new("cache", &parameter),
            //     &(i, j, k),
            //     |b, &(i, j, k)| {
            //         b.iter_with_setup(
            //             || {
            //                 let x = generate_random_matrix(i, k);
            //                 let y = generate_random_matrix(k, j);
            //                 let work_x = vec![f32::NAN; BLOCK_CACHE * BLOCK_CACHE];
            //                 let work_y = vec![f32::NAN; BLOCK_CACHE * BLOCK_CACHE];
            //                 let target = vec![f32::NAN; i * j];
            //                 (x, y, target, work_x, work_y)
            //             },
            //             |(x, y, mut target, mut work_x, mut work_y)| {
            //                 black_box(tensor_mult_cache(
            //                     &x,
            //                     &y,
            //                     &mut target,
            //                     &mut work_x,
            //                     &mut work_y,
            //                     BLOCK_CACHE,
            //                 ))
            //             },
            //         );
            //     },
            // );
        }
        group.finish();
    };
    // run_bench("MatMul - Small", &S_MATRIX_DIMS);
    // run_bench("MatMul - Medium", &M_MATRIX_DIMS);
    run_bench("MatMul - Large", &L_MATRIX_DIMS);
}
