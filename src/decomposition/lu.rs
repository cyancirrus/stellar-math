use crate::algebra::ndmethods::create_identity_matrix;
use crate::structure::ndarray::NdArray;

const TOLERANCE_CONDITION: f32 = 1e-6;

pub struct LU {
    lower: NdArray,
    upper: NdArray,
}

pub fn lu_decomposition(mut upper: NdArray) -> LU {
    // A[j, *] = c *A[i, *]
    // => c = A[i,j] / A[j,j]
    debug_assert_eq!(upper.dims[0], upper.dims[1]);
    let (rows, cols) = (upper.dims[0], upper.dims[1]); 
    let mut lower = create_identity_matrix(rows);

    for i in 0..cols {
        for j in i..rows {
            if upper.data[j * cols + i].abs() < TOLERANCE_CONDITION {
                continue;
            }
            let c = upper.data[j * cols + i] / upper.data[i * cols + i];
            lower.data[j * cols + i] = c;
            for k in i..cols {
                upper.data[j * cols + k] -= c * upper.data[i * cols + k];
            }
        }
    }
    LU { lower, upper }
}
