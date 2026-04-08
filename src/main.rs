#![allow(unused)]
// use stellar::algebra::ndmethods::{basic_mult, create_identity_matrix, tensor_mult};
// use stellar::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
// use stellar::decomposition::lq::AutumnDecomp;
// use stellar::equality::approximate::approx_vector_eq;
// use stellar::random::generation::generate_random_matrix;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use stellar::kernel::matkerns::kernel_mult;
use stellar::structure::ndarray::NdArray;

// TODO:
// then make the LX, async method

pub fn tensor_kernel(
    x: &NdArray,
    y: &NdArray,
    target: &mut [f32],
    workspace: &mut [f32],
    block: usize,
) {
    let bsize = block * block;
    let (x_rows, x_cols) = (x.dims[0], x.dims[1]);
    let y_cols = y.dims[1];
    // will reuse allocation if available
    let t_d = &mut target[..x_rows * y_cols];
    t_d.fill(0f32);
    let x_d = &x.data;
    let y_d = &y.data;
    let k_end = (x_cols + block - 1) / block;
    // debug_assert!(workspace.len() >= bsize * 2 * num_threads);
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    t_d.par_chunks_mut(block * y_cols)
        .zip(x_d.par_chunks(block * x_cols))
        .zip(workspace.par_chunks_mut(bsize * 2))
        .for_each(|((t_block_row, x_block_row), work)| {
            let (work_x, work_y) = work.split_at_mut(bsize);
            // upper threshold as i is zero indexed
            let ii_end = x_block_row.len() / x_cols;
            for k_block in 0..k_end {
                let k = k_block * block;
                let kk_end = block.min(x_cols - k);
                let mut woffset = 0;
                let mut xoffset = k;
                for _ in 0..ii_end {
                    work_x[woffset..woffset + kk_end]
                        .copy_from_slice(&x_block_row[xoffset..xoffset + kk_end]);
                    woffset += block;
                    xoffset += x_cols;
                }
                for j in (0..y_cols).step_by(block) {
                    let jj_end = block.min(y_cols - j);
                    let mut woffset = 0;
                    let mut yoffset = k * y_cols + j;
                    for _ in 0..kk_end {
                        work_y[woffset..woffset + jj_end]
                            .copy_from_slice(&y_d[yoffset..yoffset + jj_end]);
                        woffset += block;
                        yoffset += y_cols;
                    }
                    kernel_mult(
                        &work_x,
                        &work_y,
                        t_block_row,
                        ii_end,
                        kk_end,
                        jj_end,
                        y_cols,
                        j,
                    );
                }
            }
        });
}

use stellar::algebra::ndmethods::basic_mult;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;

fn test_par_equivalence() {
    let ikj = [
        (1, 1, 1),
        (8, 1, 1),
        (1, 8, 1),
        (1, 1, 8),
        (6, 4, 8),
        (6, 8, 4),
        (4, 6, 8),
        (4, 8, 6),
        (8, 4, 6),
        (8, 6, 4),
    ];
    let block = 4;
    let mut result = vec![f32::NAN; 8 * 8];
    for (i, k, j) in ikj {
        test_equivalence_mkn(block, i, k, j, &mut result);
    }
}
fn test_equivalence_mkn(block: usize, m: usize, k: usize, n: usize, result: &mut [f32]) {
    let x = generate_random_matrix(m, k);
    let y = generate_random_matrix(k, n);
    let num_threads = rayon::current_num_threads();
    let mut workspace = vec![0f32; block * block * 2 * num_threads];

    let expected = basic_mult(&x, &y);
    tensor_kernel(&x, &y, result, &mut workspace, block);
    println!("expected {expected:?}");
    println!("result {result:?}");
    assert!(approx_vector_eq(&expected.data, &result[..m * n]));
}
fn main() {
    test_par_equivalence();
}
