use crate::algebra::vector::vector_add;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;

pub fn resize_rows(m:usize, x:&mut NdArray) {
    let (rows, cols) = (x.dims[0], x.dims[1]);
    if m == rows { return; }
    else if m > rows {
        x.data.extend(vec![0_f32; (m - rows) * cols]);
    } else if m < rows {
        x.data.truncate(m * cols);
    }
    x.dims[0] = m;
}

pub fn resize_cols(n:usize, x:&mut NdArray) {
    debug_assert!(x.dims[1] > n );
    let (rows, cols) = (x.dims[0], x.dims[1]);
    if n == cols { return; }
    else if n > cols {
        x.data.extend(vec![0_f32; rows * (n - cols)]);
        for i in (0..rows).rev() {
          for j in (0..n).rev() {
              x.data.swap(i * n + j, i * cols + j);
          }
        }
    } else {
        for i in 0..rows {
          for j in 0..n {
              x.data.swap(i * n + j, i * cols + j);
          }
        }
        x.data.truncate(rows * n );
    }
    x.dims[1] = n;
}

pub fn create_identity_matrix(n: usize) -> NdArray {
    let mut data = vec![0_f32; n * n];
    let dims = vec![n; 2];
    for i in 0..n {
        data[i * n + i] = 1_f32;
    }
    NdArray { dims, data }
}
pub fn create_identity_rectangle(m:usize, n: usize) -> NdArray {
    let mut data = vec![0_f32; m * n];
    let dims = vec![m, n];
    for i in 0..m.min(n) {
        data[i * n + i] = 1_f32;
    }
    NdArray { dims, data }
}

pub fn transpose(mut ndarray: NdArray) -> NdArray {
    let rows = ndarray.dims[0];
    let cols = ndarray.dims[1];
    for i in 0..rows {
        for j in i + 1..cols {
            let temp: f32 = ndarray.data[i * rows + j];
            ndarray.data[i * rows + j] = ndarray.data[j * rows + i];
            ndarray.data[j * rows + i] = temp;
        }
    }
    ndarray.dims[0] = cols;
    ndarray.dims[1] = rows;
    ndarray
}

pub fn parallel_tensor_mult(blocksize: usize, x: &NdArray, y: &NdArray) -> NdArray {
    assert!(blocksize > 0);
    assert_eq!(x.dims[1], y.dims[0], "dimension mismatch");
    let mut dims = x.dims.clone();
    dims[1] = y.dims[1];
    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    // let y_rows = y.dims[0];
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
                                    result_block[index] += x.data[x_index] * y.data[y_index];
                                }
                            }
                        }
                    }
                    result_block
                })
                .collect::<Vec<Vec<f32>>>()
        })
        .flatten()
        .reduce(|| vec![0_f32; x_rows * y_cols], |a, b| vector_add(&a, &b));

    NdArray::new(dims, new)
}

pub fn tensor_mult(blocksize: usize, x: &NdArray, y: &NdArray) -> NdArray {
    assert!(blocksize > 0);
    assert_eq!(x.dims[1], y.dims[0], "dimension mismatch");
    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    // let y_rows = y.dims[0];
    let y_cols = y.dims[1];
    let mut new: Vec<f32> = vec![0_f32; x_rows * y_cols];
    for i in (0..x_rows).step_by(blocksize) {
        for j in (0..y_cols).step_by(blocksize) {
            for k in 0..(x_cols + blocksize - 1) / blocksize {
                for ii in 0..blocksize.min(x_rows - i) {
                    for jj in 0..blocksize.min(y_cols - j) {
                        for kk in 0..blocksize.min(x_cols - k * blocksize) {
                            let index = (i + ii) * y_cols + jj + j;
                            let x_index = (i + ii) * x_cols + k * blocksize + kk;
                            let y_index = (k * blocksize + kk) * y_cols + jj + j;
                            new[index] += x.data[x_index] * y.data[y_index];
                        }
                    }
                }
            }
        }
    }
    let mut dims = x.dims.clone();
    dims[1] = y.dims[1];
    NdArray::new(dims, new)
}
