use crate::algebra::vector::vector_add;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;

//fn multiply_summetric() {
//    //TODO: Implement sketch below
//    //
//    // for j in (0..cols).rev()
//    // for i in (j..rows).rev()
//    // for k in (0..cols) {
//    //   // store computed in the lower
//    //   data[ i * cols + j] = data[i.min(k) * cols + k.max(i)] * data[k.min(j) * cols + j.max(k)];
//    // }
//    // for i in 0..rows {
//    // for j in i+1..cols {
//    //  data[i * cols + j] = data[j * cols + i]
//    // }}
//}

pub fn create_identity_matrix(n: usize) -> NdArray {
    let mut data = vec![0f32; n * n];
    let dims = vec![n; 2];
    for i in 0..n {
        data[i * n + i] = 1f32;
    }
    NdArray { dims, data }
}
pub fn create_identity_rectangle(m: usize, n: usize) -> NdArray {
    let mut data = vec![0f32; m * n];
    let dims = vec![m, n];
    for i in 0..m.min(n) {
        data[i * n + i] = 1f32;
    }
    NdArray { dims, data }
}

pub fn parallel_tensor_mult(blocksize: usize, x: &NdArray, y: &NdArray) -> NdArray {
    // NOTE: this could use a refactor
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
                    let mut result_block: Vec<f32> = vec![0f32; x_rows * y_cols];
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
        .reduce(|| vec![0f32; x_rows * y_cols], |a, b| vector_add(&a, &b));

    NdArray::new(dims, new)
}

pub fn tensor_mult(blocksize: usize, x: &NdArray, y: &NdArray) -> NdArray {
    // should be good up until padding
    debug_assert!(blocksize > 0);
    debug_assert!(y.dims.len() > 1);
    debug_assert_eq!(x.dims[1], y.dims[0], "dimension mismatch");
    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    // let y_rows = y.dims[0];
    let y_cols = y.dims[1];
    let mut new: Vec<f32> = vec![0f32; x_rows * y_cols];
    let k_end = (x_cols + blocksize - 1) / blocksize;
    for i in (0..x_rows).step_by(blocksize) {
        let ii_end = blocksize.min(x_rows - i);
        for k in 0..k_end {
            let k_block = k * blocksize;
            let kk_end = blocksize.min(x_cols - k_block);
            for j in (0..y_cols).step_by(blocksize) {
                let jj_end = blocksize.min(y_cols - j);
                for ii in 0..ii_end {
                    let x_row = (i + ii) * x_cols;
                    let out_row = (i + ii) * y_cols;
                    for kk in 0..kk_end {
                        let k_offset = (k_block + kk) * y_cols;
                        let x_val = x.data[x_row + k_block + kk];
                        for jj in 0..jj_end {
                            new[out_row + jj + j] += x_val * y.data[k_offset + jj + j];
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

pub fn basic_mult(x: &NdArray, y: &NdArray) -> NdArray {
    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    let y_rows = y.dims[0];
    let y_cols = y.dims[1];
    assert_eq!(x_rows * x_cols, x.data.len());
    assert_eq!(y_rows * y_cols, y.data.len());
    let mut res = vec![0f32; x_rows * y_cols];

    unsafe {
        let x_ptr = x.data.as_ptr();
        let y_ptr = y.data.as_ptr();
        let res_ptr = res.as_mut_ptr();

        for i in 0..x_rows {
            let x_row = i * x_cols;
            let res_row = i * y_cols;
            for j in 0..y_cols {
                let mut sum = 0.0;
                for k in 0..x_cols {
                    sum += *x_ptr.add(x_row + k) * *y_ptr.add(k * y_cols + j);
                }
                *res_ptr.add(res_row + j) = sum;
            }
        }
    }
    NdArray::new(vec![x_rows, y_cols], res)
}

pub fn matrix_mult(x: &NdArray, y: &NdArray) -> NdArray {
    let (k, j) = (x.dims[1], y.dims[1]);
    if k <= 32 && j <= 32 {
        basic_mult(x, y)
    } else {
        tensor_mult(32, x, y)
    }
}

pub fn mult_mat_vec(a: &NdArray, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.dims[1], x.len());
    let (m, n) = (a.dims[0], a.dims[1]);
    let mut result = vec![0f32; n];
    for i in 0..m {
        for k in 0..n {
            result[i] += a.data[i * n + k] * x[k];
        }
    }
    result
}

pub fn in_place_add(x: &mut NdArray, y: &NdArray) {
    debug_assert_eq!(x.dims, y.dims);
    for i in 0..x.data.len() {
        x.data[i] += y.data[i];
    }
}

pub fn in_place_sub(x: &mut NdArray, y: &NdArray) {
    debug_assert_eq!(x.dims, y.dims);
    for i in 0..x.data.len() {
        x.data[i] -= y.data[i];
    }
}

pub fn lt_tensor_mult(blocksize: usize, x: &NdArray, y: &NdArray) -> NdArray {
    // transpose basis
    // X'Y
    debug_assert!(blocksize > 0);
    debug_assert!(y.dims.len() > 1);
    debug_assert_eq!(x.dims[0], y.dims[0], "dimension mismatch");
    let x_rows = x.dims[1];
    let x_cols = x.dims[0];
    // let y_rows = y.dims[0];
    let y_cols = y.dims[1];
    let mut new: Vec<f32> = vec![0f32; x_rows * y_cols];
    let k_end = (x_cols + blocksize - 1) / blocksize;
    for i in (0..x_rows).step_by(blocksize) {
        let ii_end = blocksize.min(x_rows - i);
        for k in 0..k_end {
            let k_block = k * blocksize;
            let kk_end = blocksize.min(x_cols - k_block);
            for j in (0..y_cols).step_by(blocksize) {
                let jj_end = blocksize.min(y_cols - j);
                for ii in 0..ii_end {
                    let out_row = (i + ii) * y_cols;
                    for kk in 0..kk_end {
                        let k_offset = (k_block + kk) * y_cols;
                        // transpose
                        let x_val = x.data[(k_block + kk) * x_rows + i + ii];
                        for jj in 0..jj_end {
                            new[out_row + jj + j] += x_val * y.data[k_offset + jj + j];
                        }
                    }
                }
            }
        }
    }
    NdArray {
        dims: vec![x_rows, y_cols],
        data: new,
    }
}

pub fn lt_basic_mult(x: &NdArray, y: &NdArray) -> NdArray {
    // transpose basis
    // X'Y
    let x_rows = x.dims[1];
    let x_cols = x.dims[0];
    let y_rows = y.dims[0];
    let y_cols = y.dims[1];
    assert_eq!(x_rows * x_cols, x.data.len());
    assert_eq!(y_rows * y_cols, y.data.len());
    let mut res = vec![0f32; x_rows * y_cols];

    unsafe {
        let x_ptr = x.data.as_ptr();
        let y_ptr = y.data.as_ptr();
        let res_ptr = res.as_mut_ptr();

        for i in 0..x_rows {
            let res_row = i * y_cols;
            for j in 0..y_cols {
                let mut sum = 0.0;
                for k in 0..x_cols {
                    sum += *x_ptr.add(k * x_rows + i) * *y_ptr.add(k * y_cols + j);
                }
                *res_ptr.add(res_row + j) = sum;
            }
        }
    }
    NdArray::new(vec![x_rows, y_cols], res)
}

pub fn lt_matrix_mult(x: &NdArray, y: &NdArray) -> NdArray {
    // X'Y
    let (k, j) = (x.dims[0], y.dims[1]);
    if k <= 32 && j <= 32 {
        lt_basic_mult(x, y)
    } else {
        lt_tensor_mult(32, x, y)
    }
}

// use stellar::random::generation::{generate_random_matrix};
// pub fn test_matrix_transpose_mult() {
//     let (m, k, n) = (2, 4, 7);
//     let x = generate_random_matrix(k, m);
//     let y = generate_random_matrix(k, n);
//     let x_t = x.transpose();

//     let expect = matrix_mult(&x_t, &y);
//     let actual = lt_matrix_mult(&x, &y);

//     println!("expect {expect:?}");
//     println!("----------------------");
//     println!("actual {actual:?}");

// }

// pub fn tensor_mult(blocksize: usize, x: &NdArray, y: &NdArray) -> NdArray {
//     assert!(blocksize > 0);
//     assert_eq!(x.dims[1], y.dims[0], "dimension mismatch");
//     let x_rows = x.dims[0];
//     let x_cols = x.dims[1];
//     // let y_rows = y.dims[0];
//     let y_cols = y.dims[1];
//     let mut new: Vec<f32> = vec![0f32; x_rows * y_cols];
//     for i in (0..x_rows).step_by(blocksize) {
//         for j in (0..y_cols).step_by(blocksize) {
//             for k in 0..(x_cols + blocksize - 1) / blocksize {
//                 for ii in 0..blocksize.min(x_rows - i) {
//                     for jj in 0..blocksize.min(y_cols - j) {
//                         for kk in 0..blocksize.min(x_cols - k * blocksize) {
//                             let index = (i + ii) * y_cols + jj + j;
//                             let x_index = (i + ii) * x_cols + k * blocksize + kk;
//                             let y_index = (k * blocksize + kk) * y_cols + jj + j;
//                             new[index] += x.data[x_index] * y.data[y_index];
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     let mut dims = x.dims.clone();
//     dims[1] = y.dims[1];
//     NdArray::new(dims, new)
// }
