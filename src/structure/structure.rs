#![allow(warnings)]
use crate::calc_utils::math;
use rayon::prelude::*;
use std::fmt;
}


}


pub fn parallel_tensor_mult(blocksize: usize, x: &NdArray, y: &NdArray) -> NdArray {
    assert!(blocksize > 0);
    assert_eq!(x.dims[1], y.dims[0], "dimension mismatch");
    let mut dims = x.dims.clone();
    dims[1] = y.dims[1];
    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    let y_rows = y.dims[0];
    let y_cols = y.dims[1];

    // iterate by blocksize
    let new = (0..x_rows)
        .step_by(blocksize)
        .collect::<Vec<usize>>()
        .into_par_iter()
        .map(|i| {
            (0..y_cols)
                .step_by(blocksize)
                .map(|j| {
                    let mut result_block: Vec<f32> = vec![0_f32; x_rows * y_cols];
                    for k in 0..(x_cols + blocksize - 1) / blocksize {
                        for ii in 0..blocksize.min(x_rows - i) {
                            for jj in 0..blocksize.min(y_cols - j) {
                                for kk in 0..blocksize.min(x_cols - k * blocksize) {
                                    let index = (i + ii) * y_cols + jj + j;
                                    let x_index = (i + ii) * x_cols + k * blocksize + kk;
                                    let y_index = (k * blocksize + kk) * y_cols + jj + j;
                                    result_block[index] += { x.data[x_index] * y.data[y_index] };
                                }
                            }
                        }
                    }
                    result_block
                })
                .collect::<Vec<Vec<f32>>>()
        })
        .flatten()
        .reduce(
            || vec![0_f32; x_rows * y_cols],
            |a, b| math::vector_add(&a, &b),
        );

    NdArray::new(dims, new)
}
