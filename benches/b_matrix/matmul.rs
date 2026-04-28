#![allow(unused_imports)]
use crate::sharedvars::{L_MATRIX_DIMS, M_MATRIX_DIMS, S_MATRIX_ALIGNED, S_MATRIX_UNALIGNED, L_MAT10S_DIMS, M_MAT10_DIMS};
use criterion::{BenchmarkId, Criterion, Throughput};
use faer::linalg::matmul::matmul;
use faer::prelude::*;
use ndarray::Array2;
use std::hint::black_box;
use stellar::algebra::bmethods::tensor_kernel_new;
use stellar::algebra::mmethods::tensor_kernel;
use stellar::algebra::ndmethods::{basic_mult, tensor_mult};
use stellar::random::generation::generate_random_matrix;
// use criterion::{AxisScale, PlotConfiguration};

pub fn bench_matmul_scaling(c: &mut Criterion) {
    let mut run_bench = |group_name: &str, dims: &[(usize, usize, usize)]| {
        let mut group = c.benchmark_group(group_name);
        group.sampling_mode(criterion::SamplingMode::Auto);
        for &(i, k, j) in dims {
            let parameter = format!("{}x{}x{}", i, k, j);
            group.throughput(Throughput::Elements((2 * i * k * j) as u64));
            group.bench_with_input(
                BenchmarkId::new("saturated_kernel", &parameter),
                &(i, j, k),
                |b, &(i, j, k)| {
                    b.iter_with_setup(
                        || {
                            let x = generate_random_matrix(i, k);
                            let y = generate_random_matrix(k, j);
                            let target = vec![f32::NAN; i * j];
                            (x, y, target)
                        },
                        |(x, y, mut target)| black_box(tensor_kernel_new(&x, &y, &mut target)),
                    )
                },
            );
            // group.bench_with_input(
            //     BenchmarkId::new("tensor_kernel", &parameter),
            //     &(i, j, k),
            //     |b, &(i, j, k)| {
            //         b.iter_with_setup(
            //             || {
            //                 let x = generate_random_matrix(i, k);
            //                 let y = generate_random_matrix(k, j);
            //                 let target = vec![f32::NAN; i * j];
            //                 (x, y, target)
            //             },
            //             |(x, y, mut target)| black_box(tensor_kernel(&x, &y, &mut target)),
            //         )
            //     },
            // );
            // group.bench_with_input(
            //     BenchmarkId::new("faer", &parameter),
            //     &(i, j, k),
            //     |b, &(m, n, k)| {
            //         b.iter_with_setup(
            //             || {
            //                 let data_x = generate_random_matrix(m, k);
            //                 let data_y = generate_random_matrix(k, n);
            //                 let x = faer::Mat::from_fn(m, k, |r, c| data_x.data[r * k + c]);
            //                 let y = faer::Mat::from_fn(k, n, |r, c| data_y.data[r * n + c]);
            //                 let target = faer::Mat::<f32>::zeros(m, n);
            //                 (x, y, target)
            //             },
            //             |(x, y, mut target)| {
            //                 let threads = rayon::current_num_threads();
            //                 matmul(
            //                     target.as_mut(),
            //                     faer::Accum::Replace,
            //                     x.as_ref(),
            //                     y.as_ref(),
            //                     1.0f32,
            //                     faer::Par::Rayon(std::num::NonZero::new(threads).unwrap()),
            //                 );
            //                 black_box(target)
            //             },
            //         );
            //     },
            // );
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
            //             |(x, y)| black_box(x.dot(&y)),
            //         );
            //     },
            // );
            // group.bench_with_input(
            //     BenchmarkId::new("tensor_parkernel", &parameter),
            //     &(i, j, k),
            //     |b, &(i, j, k)| {
            //         b.iter_with_setup(
            //             || {
            //                 let x = generate_random_matrix(i, k);
            //                 let y = generate_random_matrix(k, j);
            //                 let target = vec![f32::NAN; i * j];
            //                 (x, y, target)
            //             },
            //             |(x, y, mut target)| black_box(tensor_parkern(&x, &y, &mut target)),
            //         )
            //     },
            // );
            // group.throughput(Throughput::Elements((2 * i * k * j) as u64));
            // group.bench_with_input(
            //     BenchmarkId::new("tensor_minikern", &parameter),
            //     &(i, j, k),
            //     |b, &(i, j, k)| {
            //         b.iter_with_setup(
            //             || {
            //                 let x = generate_random_matrix(i, k);
            //                 let y = generate_random_matrix(k, j);
            //                 let target = vec![f32::NAN; i * j];
            //                 (x, y, target)
            //             },
            //             |(x, y, mut target)| black_box(tensor_minikern(&x, &y, &mut target)),
            //         )
            //     },
            // );
        }
        group.finish();
    };
    // run_bench("MatMul - Small", &S_MATRIX_ALIGNED);
    // run_bench("MatMul - Small Unaligned", &S_MATRIX_UNALIGNED);
    // run_bench("MatMul - Medium", &M_MATRIX_DIMS);
    // run_bench("MatMul - Large", &L_MATRIX_DIMS);
    run_bench("MatMul - Medium", &M_MATRIX_DIMS);
    run_bench("MatMul - Large", &L_MATRIX_DIMS);
    run_bench("MatMul - Medium", &M_MAT10_DIMS);
    run_bench("MatMul - Large", &L_MAT10S_DIMS);
}
