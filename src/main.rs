#![allow(unused)]
use stellar::algebra::ndmethods::{basic_mult, create_identity_matrix, tensor_mult};
use stellar::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

fn tensor_mult_cache(
    x: &NdArray,
    y: &NdArray,
    target: &mut [f32],
    work_x: &mut [f32],
    work_y: &mut [f32],
    block: usize,
) {
    let bsize = block * block;
    let work_x = &mut work_x[..bsize];
    let work_y = &mut work_y[..bsize];
    let (x_rows, x_cols) = (x.dims[0], x.dims[1]);
    let y_cols = y.dims[1];
    debug_assert!(work_x.len() >= bsize);
    debug_assert!(work_y.len() >= bsize);
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    // will reuse allocation if available
    let target = &mut target[..x_rows * y_cols];
    target.fill(0f32);
    let x_d = &x.data;
    let y_d = &y.data;
    let k_end = (x_cols + block - 1) / block;
    for i in (0..x_rows).step_by(block) {
        // upper threshold as i is zero indexed
        let ii_end = block.min(x_rows - i);
        for k_block in 0..k_end {
            let k = k_block * block;
            // let kk_end = block.min(x_cols - k);
            let kk_end = block.min(x_cols - k);
            let mut woffset = 0;
            let mut xoffset = i * x_cols + k;
            for _ in 0..ii_end {
                work_x[woffset..woffset + kk_end].copy_from_slice(&x_d[xoffset..xoffset + kk_end]);
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
                for ii in 0..ii_end {
                    let x_row = ii * block;
                    let out_row = (i + ii) * y_cols;
                    for kk in 0..kk_end {
                        let k_offset = kk * block;
                        let x_val = work_x[x_row + kk];
                        for jj in 0..jj_end {
                            target[out_row + jj + j] += x_val * work_y[k_offset + jj];
                        }
                    }
                }
            }
        }
    }
}
fn tmat_mult_left_lower(
    x: &NdArray,
    y: &NdArray,
    target:&mut NdArray,
    work_x: &mut [f32],
    work_y: &mut [f32],
    block: usize,
) {
    // should be good up until padding
    debug_assert!(block > 0);
    debug_assert!(y.dims.len() > 1);
    debug_assert!(work_x.len() > block * block);
    debug_assert!(work_y.len() > block * block);
    debug_assert_eq!(x.dims[1], y.dims[0], "dimension mismatch");
    let (x_rows, x_cols) = (x.dims[0], x.dims[1]);
    let y_cols = y.dims[1];
    // will reuse allocation if available
    target.resize(x_rows, y_cols);
    let t = &mut target.data;
    let x_d = &x.data;
    let y_d = &y.data;
    for i in (0..x_rows).step_by(block) {
        // upper threshold as i is zero indexed
        let k_end = (i + block) / block;
        let ii_end = block.min(x_rows - i);
        for k_block in 0..k_end {
            let k = k_block * block;
            let kk_end = block.min(x_cols - k);
            let mut woffset = 0;
            let mut xoffset = i * x_cols + k;
            for _ in 0..block {
                work_x[woffset..woffset + block].copy_from_slice(&x_d[xoffset..xoffset + kk_end]);
                woffset += block;
                xoffset += x_cols;
            }
            if i == k {
                let mut doffset = 0;
                // handle the case when blocks are on the diagonal
                for d in 1..block {
                    // fills to the right of the diagonal
                    work_x[doffset + d..doffset + block].fill(0f32);
                    doffset += block;
                }
            }
            for j in (0..y_cols).step_by(block) {
                let jj_end = block.min(y_cols - j);
                let mut woffset = 0;
                let mut yoffset = k * y_cols + j;
                for _ in 0..block {
                    work_y[woffset..woffset + jj_end]
                        .copy_from_slice(&y_d[yoffset..yoffset + jj_end]);
                    woffset += block;
                    yoffset += y_cols;
                }
                for ii in 0..ii_end {
                    let x_row = ii * block;
                    let out_row = (i + ii) * y_cols;
                    for kk in 0..kk_end {
                        let k_offset = kk * block;
                        let x_val = work_x[x_row + kk];
                        for jj in 0..jj_end {
                            t[out_row + jj + j] += x_val * work_y[k_offset + jj];
                        }
                    }
                }
            }
        }
    }
}

fn test_equivalence() {
    // cols >= rows
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
    let mut work_x = vec![f32::NAN; block * block];
    let mut work_y = vec![f32::NAN; block * block];
    let mut result = vec![f32::NAN; 8 * 8];
    for (i, k, j) in ikj {
        test_equivalence_mkn(block, i, k, j, &mut work_x, &mut work_y, &mut result);
    }
}
fn test_equivalence_mkn(block:usize, m:usize, k:usize, n:usize, work_x:&mut [f32], work_y:&mut [f32], result: &mut [f32]) {
    let x = generate_random_matrix(m, k);
    let y = generate_random_matrix(k, n);
    
    let expected = basic_mult( &x, &y);
    tensor_mult_cache(&x, &y, result, work_x, work_y, block);
    println!("result {result:?}");
    assert!(approx_vector_eq(&expected.data, &result[..m * n]));
}


fn main() {
    test_equivalence();
}
