use crate::algebra::ndmethods::create_identity_matrix;
use crate::structure::ndarray::NdArray;

const TOLERANCE_CONDITION: f32 = 1e-6;

pub struct LU {
    pub lower: NdArray,
    pub upper: NdArray,
}

// pub fn lu_decompose(mut upper: NdArray) -> LU {
//     // A[j, *] = c *A[i, *]
//     // => c = A[i,j] / A[j,j]
//     // could be extended to non-square matrices 
//     debug_assert_eq!(upper.dims[0], upper.dims[1]);
//     let (rows, cols) = (upper.dims[0], upper.dims[1]); 
//     let mut lower = create_identity_matrix(rows);

//     for i in 0..cols {
//         for j in i..rows {
//             if upper.data[j * cols + i].abs() < TOLERANCE_CONDITION {
//                 continue;
//             }
//             let c = upper.data[j * cols + i] / upper.data[i * cols + i];
//             lower.data[j * cols + i] = c;
//             for k in i..cols {
//                 upper.data[j * cols + k] -= c * upper.data[i * cols + k];
//             }
//         }
//     }
//     LU { lower, upper }
// }

pub fn lu_decompose(mut upper: NdArray) -> LU {
    // A[j, *] = c *A[i, *]
    // => c = A[i,j] / A[j,j]
    // could be extended to non-square matrices 
    debug_assert_eq!(upper.dims[0], upper.dims[1]);
    let (rows, cols) = (upper.dims[0], upper.dims[1]); 
    let mut lower = create_identity_matrix(rows);

    for i in 0..rows {
       for j in 0..cols {
           if i == j {
                for k in 0..i {
                    upper.data[ i * cols  + j] -= lower.data[i * cols + k] * upper.data[k * cols + j]
                }
           } else {
                for k in 0.. i {
                    upper.data[i * cols + j] -= lower.data[i * cols + k] * upper.data[k * cols + j]
                }
                if i > j {
                    lower.data[ i * cols + j ] = upper.data[ i * cols + j ] /  upper.data[ j * cols + j ]
                } else {
                    upper.data[ i * cols + j ] = upper.data[ i * cols + j] / lower.data[ i * cols + i]
                }
           }
       }
    }
    for i in 0..rows {
        for j in 0..i {
            upper.data[ i * cols + j ] = 0_f32;
        }
    }
    LU { lower, upper }
}


    // for i in 0..rows {
    //    for j in 0..cols {
    //        if i == j {
    //             u[i,j] = a[i,i];
    //             l[i,j] = 1_f32;
    //        } else {
    //             for k in 0.. i {
    //                 a[i,j] -= l[i, 0] * u[0, j]
    //             }
    //             if i > j {
    //                 l[i,j] = a[i,j]/u[j,j]
    //             } else {
    //                 u[i,j] = a[i,j]/l[i,i]
    //             }
    //        }
    //    }
    // }
