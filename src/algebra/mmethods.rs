use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;

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

pub fn par_tensor_mult_cache(x: &NdArray, y: &NdArray, target: &mut [f32], block: usize) {
    let bsize = block * block;
    let (x_rows, x_cols) = (x.dims[0], x.dims[1]);
    let y_cols = y.dims[1];
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    // will reuse allocation if available
    let t_d = &mut target[..x_rows * y_cols];
    t_d.fill(0f32);
    let x_d = &x.data;
    let y_d = &y.data;
    let k_end = (x_cols + block - 1) / block;
    t_d.par_chunks_mut(block * y_cols)
        .zip(x_d.par_chunks(block * x_cols))
        .for_each(|(t_block_row, x_block_row)| {
            let mut work_x = vec![0f32; bsize];
            let mut work_y = vec![0f32; bsize];
            // upper threshold as i is zero indexed
            let ii_end = x_block_row.len() / x_cols;
            for k_block in 0..k_end {
                let k = k_block * block;
                let kk_end = block.min(x_cols - k);
                let mut woffset = 0;
                // let mut xoffset = i * x_cols + k;
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
                    for ii in 0..ii_end {
                        // index into the work_x
                        let x_row = ii * block;
                        let out_row = ii * y_cols;
                        for kk in 0..kk_end {
                            let k_offset = kk * block;
                            let x_val = work_x[x_row + kk];
                            for jj in 0..jj_end {
                                t_block_row[out_row + jj + j] += x_val * work_y[k_offset + jj];
                            }
                        }
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

        let expected = basic_mult(&x, &y);
        par_tensor_mult_cache(&x, &y, result, block);
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
}
