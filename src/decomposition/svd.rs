use crate::algebra::ndmethods::{
    create_identity_matrix, resize_cols, resize_rows, tensor_mult, transpose,
};
use crate::algebra::vector::{initialize_unit_vector, magnitude};
use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;

// TODO: store the bidiagonal

// halko-trop
pub fn golub_kahan(mut a: NdArray) -> NdArray {
    // singular vlaues are up to sign convention
    // ie if l[k] < 0 => sign flip u[k]
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let card = rows.min(cols) - (rows <= cols) as usize;
    let mut proj: HouseholderReflection;
    let mut w = vec![0f32; rows.max(cols)];
    for o in 0..card {
        proj = householder_params(
            // column vector
            (o..rows)
                .into_iter()
                .map(|r| a.data[r * cols + o])
                .collect(),
        );
        // (I - bvv')A => w := bv'A
        //  A -= vw'
        for j in o..cols {
            for i in o..rows {
                w[j] += proj.vector[i - o] * a.data[i * cols + j];
            }
            w[j] *= proj.beta;
            for i in o..rows {
                a.data[i * cols + j] -= proj.vector[i - o] * w[j];
            }
            w[j] = 0_f32;
        }
        // stop one early for columns because a[m,n-1] can be non-zero
        if o + 1 == card {
            break;
        }
        let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
        proj = householder_params(
            // row vector
            a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec(),
        );
        // A(I - bvv') => w = b * Av
        // A -= wv'
        for i in o..rows {
            for j in o + 1..cols {
                w[i] += a.data[i * cols + j] * proj.vector[j - o - 1];
            }
            w[i] *= proj.beta;
            for j in o + 1..cols {
                a.data[i * cols + j] -= w[i] * proj.vector[j - o - 1];
            }
            w[i] = 0_f32;
        }
    }
    let dims = rows.min(cols);
    resize_rows(dims, &mut a);
    resize_cols(dims, &mut a);
    a
}

pub fn full_golub_kahan(mut a: NdArray) -> (NdArray, NdArray, NdArray) {
    // singular vlaues are up to sign convention
    // ie if l[k] < 0 => sign flip u[k]
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let card = rows.min(cols) - (rows <= cols) as usize;
    let mut u = create_identity_matrix(rows);
    let mut v = create_identity_matrix(cols);
    let mut proj: HouseholderReflection;
    let mut w = vec![0f32; rows.max(cols)];
    for o in 0..card {
        proj = householder_params(
            // column vector
            (o..rows).map(|r| a.data[r * cols + o]).collect(),
        );
        // (I - bvv')A => w := bv'A
        //  A -= vw'
        for j in o..cols {
            for i in o..rows {
                w[j] += proj.vector[i - o] * a.data[i * cols + j];
            }
            w[j] *= proj.beta;
            for i in o..rows {
                a.data[i * cols + j] -= proj.vector[i - o] * w[j];
            }
            w[j] = 0_f32;
        }
        for i in 0..rows {
            let mut sum = 0f32;
            for k in o..rows {
                sum += u.data[i * rows + k] * proj.vector[k - o];
            }
            sum *= proj.beta;
            for k in o..rows {
                u.data[i * rows + k] -= sum * proj.vector[k - o];
            }
        }
        // stop one early for columns because a[m,n-1] can be non-zero
        if o + 1 == card {
            break;
        }
        let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
        proj = householder_params(
            // row vector
            a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec(),
        );
        // A(I - bvv') => w = b * Av
        // A -= wv'
        for i in o..rows {
            for j in o + 1..cols {
                w[i] += a.data[i * cols + j] * proj.vector[j - o - 1];
            }
            w[i] *= proj.beta;
            for j in o + 1..cols {
                a.data[i * cols + j] -= w[i] * proj.vector[j - o - 1];
            }
            w[i] = 0_f32;
        }
        for j in 0..cols {
            let mut sum = 0_f32;
            for k in o + 1..cols {
                sum += v.data[j * cols + k] * proj.vector[k - o - 1];
            }
            sum *= proj.beta;
            for k in o + 1..cols {
                v.data[j * cols + k] -= sum * proj.vector[k - o - 1];
            }
        }
    }
    let dims = rows.min(cols);
    resize_rows(dims, &mut a);
    resize_cols(dims, &mut a);
    (u, a, v)
}

// pub fn full_golub_kahan(mut a: NdArray) -> (NdArray, NdArray, NdArray) {
//     // singular vlaues are up to sign convention
//     // ie if l[k] < 0 => sign flip u[k]
//     let (rows, cols) = (a.dims[0], a.dims[1]);
//     let card = rows.min(cols) - (rows <= cols) as usize;
//     let mut u = create_identity_matrix(rows);
//     let mut v = create_identity_matrix(cols);
//     let mut proj: HouseholderReflection;
//     let mut w = vec![0f32; rows.max(cols)];
//     for o in 0..card {
//         proj = householder_params(
//             // column vector
//             (o..rows)
//                 .into_iter()
//                 .map(|r| a.data[r * cols + o])
//                 .collect(),
//         );
//         // (I - bvv')A => w := bv'A
//         //  A -= vw'
//         for j in o..cols {
//             for i in o..rows {
//                 w[j] += proj.vector[i - o] * a.data[i * cols + j];
//             }
//             w[j] *= proj.beta;
//             for i in o..rows {
//                 a.data[i * cols + j] -= proj.vector[i - o] * w[j];
//             }
//             w[j] = 0_f32;
//         }
//         // stop one early for columns because a[m,n-1] can be non-zero
//         if o + 1 == card {
//             break;
//         }
//         let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
//         proj = householder_params(
//             // row vector
//             a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec(),
//         );
//         // A(I - bvv') => w = b * Av
//         // A -= wv'
//         for i in o..rows {
//             for j in o + 1..cols {
//                 w[i] += a.data[i * cols + j] * proj.vector[j - o - 1];
//             }
//             w[i] *= proj.beta;
//             for j in o + 1..cols {
//                 a.data[i * cols + j] -= w[i] * proj.vector[j - o - 1];
//             }
//             w[i] = 0_f32;
//         }
//         for i in 0..rows {
//             let mut sum = 0f32;
//             for k in o..rows {
//                 sum += u.data[k * rows + i] * proj.vector[i - o];
//             }
//             sum *= proj.beta;
//             for k in o..rows {
//                 u.data[i * rows + k] -= sum * proj.vector[i - o];
//             }
//         }
//         for j in 0..cols {
//             let mut sum = 0_f32;
//             for k in o+1..cols {
//                 sum += v.data[j * cols  + k ] * proj.vector[k - o - 1];
//             }
//             sum *= proj.beta;
//             for k in o+1..cols {
//                 sum += v.data[j * cols  + k ] * proj.vector[k - o - 1];
//             }
//         }
//     }
//     let dims = rows.min(cols);
//     resize_rows(dims, &mut a);
//     resize_cols(dims, &mut a);
//     (u, a, v)
// }

// NOTE: keepping around until more testing is provided
// pub fn golub_kahan(mut a: NdArray) -> NdArray {
//     let (rows, cols)  = (a.dims[0], a.dims[1]);
//     let card = rows.min(cols) - (rows <= cols) as usize;
//     let mut householder: HouseholderReflection;

//     let mut new: NdArray;
//     for o in 0..card {
//         new = create_identity_matrix(rows);
//         let column_vector = (o..rows)
//             .into_par_iter()
//             .map(|r| a.data[r * cols + o])
//             .collect::<Vec<f32>>();
//         householder = householder_params(column_vector);
//         if householder.beta < EPSILON {
//             continue;
//         }
//         for i in 0..rows - o {
//             for j in 0..cols - o {
//                 new.data[(o + i) * cols + j + o] -=
//                     householder.beta * householder.vector[i] * householder.vector[j]
//             }

//         }
//         a = tensor_mult(4, &new, &a);
//         if o < card - 1 {
//             new = create_identity_matrix(cols);
//             // let row_vector: Vec<f32> = a.data[(o * cols) + 1..(o + 1) * cols].to_vec();
//             let row_vector: Vec<f32> = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
//             householder = householder_params(row_vector);

//             for i in 0..rows - o - 1 {
//                 for j in 0..cols - o - 1 {
//                     new.data[(o + i + 1) * cols + (j + o + 1)] -=
//                         householder.beta * householder.vector[i] * householder.vector[j];
//                 }
//             }
//             a = tensor_mult(4, &a, &new);
//         }
//     }
//     a
// }
