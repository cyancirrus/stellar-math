use crate::arch::SIMD_WIDTH;
use crate::kernel::matkerns::kernel_mult;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;

pub fn tensor_kernel(x: &NdArray, y: &NdArray, target: &mut [f32], workspace: &mut [f32]) {
    let bsize = SIMD_WIDTH * SIMD_WIDTH;
    let (x_rows, x_cols) = (x.dims[0], x.dims[1]);
    let y_cols = y.dims[1];
    // will reuse allocation if available
    let t_d = &mut target[..x_rows * y_cols];
    t_d.fill(0f32);
    let x_d = &x.data;
    let y_d = &y.data;
    let k_end = (x_cols + SIMD_WIDTH - 1) / SIMD_WIDTH;
    // debug_assert!(workspace.len() >= bsize * 2 * num_threads);
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    t_d.par_chunks_mut(SIMD_WIDTH * y_cols)
        .zip(x_d.par_chunks(SIMD_WIDTH * x_cols))
        .zip(workspace.par_chunks_mut(bsize * 2))
        .for_each(|((t_block_row, x_block_row), work)| {
            let (work_x, _) = work.split_at_mut(bsize);
            // upper threshold as i is zero indexed
            let ii_end = x_block_row.len() / x_cols;
            let mut k = 0;
            for _ in 0..k_end {
                let kk_end = SIMD_WIDTH.min(x_cols - k);
                let mut woffset = 0;
                let mut xoffset = k;
                let mut yoffset = k * y_cols;
                // kernel methods where need 0 are handled with iterator
                for _ in 0..ii_end {
                    work_x[woffset..woffset + kk_end]
                        .copy_from_slice(&x_block_row[xoffset..xoffset + kk_end]);
                    woffset += SIMD_WIDTH;
                    xoffset += x_cols;
                }
                for j in (0..y_cols).step_by(SIMD_WIDTH) {
                    let jj_end = SIMD_WIDTH.min(y_cols - j);
                    let y_align = &y_d[yoffset..yoffset + (kk_end - 1) * y_cols + jj_end];
                    let t_align = &mut t_block_row[j..];
                    kernel_mult(
                        &work_x, y_align, t_align, ii_end, kk_end, jj_end, SIMD_WIDTH, y_cols,
                    );
                    yoffset += SIMD_WIDTH;
                }
                k += SIMD_WIDTH;
            }
        });
}

pub fn tensor_mult_cache(
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
                        // let k_offset = kk * block;
                        // let x_val = work_x[x_row + kk];
                        // for jj in 0..jj_end {
                        //     target[out_row + jj + j] += x_val * work_y[k_offset + jj];
                        // }
                        let k_offset = kk * block;
                        let x_val = work_x[x_row + kk];
                        let t_select = &mut target[out_row + j..out_row + j + jj_end];
                        let y_select = &work_y[k_offset..k_offset + jj_end];
                        for (t, y) in t_select.iter_mut().zip(y_select.iter()) {
                            *t += x_val * y;
                        }
                    }
                }
            }
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
            test_kernel_equivalence_mkn(SIMD_WIDTH, i, k, j, &mut result);
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
    #[test]
    fn test_equivalence() {
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
    fn test_par_equivalence_mkn(block: usize, m: usize, k: usize, n: usize, result: &mut [f32]) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);
        let num_threads = rayon::current_num_threads();
        let mut workspace = vec![0f32; block * block * 2 * num_threads];

        let expected = basic_mult(&x, &y);
        par_tensor_mult_cache(&x, &y, result, &mut workspace, block);
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    fn test_equivalence_mkn(
        block: usize,
        m: usize,
        k: usize,
        n: usize,
        work_x: &mut [f32],
        work_y: &mut [f32],
        result: &mut [f32],
    ) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);

        let expected = basic_mult(&x, &y);
        tensor_mult_cache(&x, &y, result, work_x, work_y, block);
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    fn test_kernel_equivalence_mkn(block: usize, m: usize, k: usize, n: usize, result: &mut [f32]) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);
        let num_threads = rayon::current_num_threads();
        let mut workspace = vec![0f32; block * block * 2 * num_threads];

        let expected = basic_mult(&x, &y);
        tensor_kernel(&x, &y, result, &mut workspace);
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
}
