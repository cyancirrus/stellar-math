use crate::algebra::ndmethods::create_identity_matrix;
use crate::structure::ndarray::NdArray;

const TOLERANCE_CONDITION: f32 = 1e-6;

// TODO: Consider making this densely packed
pub struct LU {
    // This could be densely packed in order to save memory
    // However would need virtual methods to multiply component parts
    pub lower: NdArray,
    pub upper: NdArray,
}

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
