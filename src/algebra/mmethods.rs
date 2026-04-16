use crate::arch::SIMD_WIDTH;
use crate::kernel::matkerns::kernel_mult;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;

thread_local! {
    static PROC_WORKSPACE: RefCell<Vec<f32>> = RefCell::new(vec![0.0f32; SIMD_WIDTH * SIMD_WIDTH]);
}

pub fn tensor_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    unsafe {
        // will reuse allocation if available
        let x_d = &x.data;
        let y_d = &y.data;
        let (x_rows, x_cols) = (x.dims[0], x.dims[1]);
        let y_cols = y.dims[1];
        let t_d = &mut target[..x_rows * y_cols];
        t_d.fill(0f32);
        debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
        t_d.par_chunks_mut(SIMD_WIDTH * y_cols)
            .zip(x_d.par_chunks(SIMD_WIDTH * x_cols))
            .for_each(|(t_block_row, x_block_row)| {
                PROC_WORKSPACE.with(|workspace_cell| {
                    let ii_end = x_block_row.len() / x_cols;
                    let mut work_x = workspace_cell.borrow_mut();
                    let mut yoffset = 0;
                    for k in (0..x_cols).step_by(SIMD_WIDTH) {
                        let kk_end = SIMD_WIDTH.min(x_cols - k);
                        let mut xoffset = k;
                        let mut woffset = 0;
                        for _ in 0..ii_end {
                            work_x
                                .get_unchecked_mut(woffset..woffset + kk_end)
                                .copy_from_slice(
                                    &x_block_row.get_unchecked(xoffset..xoffset + kk_end),
                                );
                            woffset += SIMD_WIDTH;
                            xoffset += x_cols;
                        }
                        for j in (0..y_cols).step_by(SIMD_WIDTH) {
                            let jj_end = SIMD_WIDTH.min(y_cols - j);
                            kernel_mult(
                                &work_x,
                                y_d.get_unchecked(yoffset + j..),
                                t_block_row.get_unchecked_mut(j..),
                                ii_end,
                                kk_end,
                                jj_end,
                                SIMD_WIDTH,
                                y_cols,
                            );
                        }
                        yoffset += SIMD_WIDTH * y_cols;
                    }
                })
            });
    }
}

pub fn tensor_minikern(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    unsafe {
        // will reuse allocation if available
        let x_d = &x.data;
        let y_d = &y.data;
        let (x_rows, x_cols) = (x.dims[0], x.dims[1]);
        let y_cols = y.dims[1];
        let t_d = &mut target[..x_rows * y_cols];
        t_d.fill(0f32);
        debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
        let mut xoffset = 0;
        let mut toffset = 0;
        for i in (0..x_rows).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(x_rows - i);
            let mut yoffset = 0;
            for k in (0..x_cols).step_by(SIMD_WIDTH) {
                let kk_end = SIMD_WIDTH.min(x_cols - k);
                for j in (0..y_cols).step_by(SIMD_WIDTH) {
                    let jj_end = SIMD_WIDTH.min(y_cols - j);
                    kernel_mult(
                        x_d.get_unchecked(xoffset + k..),
                        y_d.get_unchecked(yoffset + j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        ii_end,
                        kk_end,
                        jj_end,
                        x_cols,
                        y_cols,
                    );
                }
                yoffset += SIMD_WIDTH * y_cols;
            }
            toffset += SIMD_WIDTH * y_cols;
            xoffset += SIMD_WIDTH * x_cols;
        }
    }
}

pub fn par_tensor_mult_cache(
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
            // wrong
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
                    let mut x_row = 0;
                    let mut out_row = 0;
                    for _ in 0..ii_end {
                        // index into the work_x
                        let mut koffset = 0;
                        let mut x_idx = x_row;
                        for _ in 0..kk_end {
                            let x_val = work_x[x_idx];
                            let t_slice = &mut t_block_row[out_row + j..out_row + j + jj_end];
                            let y_slice = &work_y[koffset..koffset + jj_end];
                            // for (t, y) in t_slice.iter_mut().zip(y_slice.iter()) {
                            //     *t += x_val * y;
                            // }
                            unsafe {
                                for idx in 0..jj_end {
                                    *t_slice.get_unchecked_mut(idx) +=
                                        x_val * *y_slice.get_unchecked(idx);
                                }
                            }
                            koffset += block;
                            x_idx += 1;
                        }
                        x_row += block;
                        out_row += y_cols;
                    }
                }
            }
        });
}

#[cfg(test)]
mod test_cached_matrix_methods {
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;

    #[test]
    // #[cfg(feature = "avx2")]
    fn test_kernel_equivalence() {
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
            (16, 8, 16),
        ];
        let mut result = vec![f32::NAN; 16 * 16];
        for (i, k, j) in ikj {
            test_kernel_equivalence_mkn(i, k, j, &mut result);
        }
    }
    #[test]
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
            test_par_equivalence_mkn(block, i, k, j, &mut result);
        }
    }
    fn test_par_equivalence_mkn(block: usize, m: usize, k: usize, n: usize, result: &mut [f32]) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);
        let num_threads = rayon::current_num_threads();
        let mut workspace = vec![0f32; block * block * 2 * num_threads];

        let expected = basic_mult(&x, &y);
        par_tensor_mult_cache(&x, &y, result, &mut workspace, block);
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    fn test_kernel_equivalence_mkn(m: usize, k: usize, n: usize, result: &mut [f32]) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);
        let expected = basic_mult(&x, &y);
        tensor_kernel(&x, &y, result);
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    #[test]
    fn test_minikern_equivalence() {
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
            test_minikern_equivalence_mkn(block, i, k, j, &mut result);
        }
    }
    fn test_minikern_equivalence_mkn(
        block: usize,
        m: usize,
        k: usize,
        n: usize,
        result: &mut [f32],
    ) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);
        let mut result = vec![0f32; m * n];

        let expected = basic_mult(&x, &y);
        tensor_minikern(&x, &y, &mut result);
        let inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
}
