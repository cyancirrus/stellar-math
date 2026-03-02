#![allow(unused_imports)]
use std::hint::black_box;
use std::time::{Duration, Instant};
use stellar::algebra::ndmethods::{matrix_mult, tensor_mult};
use stellar::random::generation::generate_random_matrix;
use stellar::solver::randomized_svd::RandomizedSvd;
use stellar::structure::ndarray::NdArray;
use rayon::prelude::*;

pub fn par_tensor_mult(blocksize: usize, x: &NdArray, y: &NdArray) -> NdArray {
    // should be good up until padding
    debug_assert!(blocksize > 0);
    debug_assert!(y.dims.len() > 1);
    debug_assert_eq!(x.dims[1], y.dims[0], "dimension mismatch");
    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    // let y_rows = y.dims[0];
    let y_cols = y.dims[1];
    let mut data: Vec<f32> = vec![0f32; x_rows * y_cols];
    data.par_chunks_mut(blocksize * y_cols)
        .zip(x.data.par_chunks(blocksize * x_cols))
        .for_each( |(data_block, x_block)| {
            let ii_end = data_block.len() / y_cols;
            for k in (0..x_cols).step_by(blocksize) {
                let k_block = (k + blocksize).min(x_cols);
                for j in (0..y_cols).step_by(blocksize) {
                    let j_offset = (j + blocksize).min(y_cols);
                    for ii in 0..ii_end {
                        let local_x_row = ii * x_cols;
                        let local_out_row = ii * y_cols;
                        for kk in k..k_block {
                            let x_val = x_block[local_x_row + kk];
                            let k_offset = kk  * y_cols;
                            let out_row = &mut data_block[local_out_row + j..local_out_row + j_offset];
                            let y_slice = &y.data[k_offset + j..k_offset + j_offset];
                            for (o, y) in out_row.iter_mut().zip(y_slice.iter()) {
                                *o += x_val * y;
                            }
                        }
                    }
                }
            }
        }
    );
    NdArray { dims:vec![x.dims[0], y.dims[1]], data }
}

fn test_p_mult() {
    let (m, k, n) = (6, 4, 2);
    let (x, y) = (generate_random_matrix(m, k), generate_random_matrix(k, n));
    let expect = tensor_mult(4, &x, &y);
    let actual = par_tensor_mult(4, &x, &y);
    
    println!("expect {expect:?}");
    println!("---------------------");
    println!("actual {actual:?}");

}


fn main() {
    test_p_mult();
}
