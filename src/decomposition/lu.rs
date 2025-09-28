use crate::algebra::ndmethods::create_identity_matrix;
use crate::structure::ndarray::NdArray;

const TOLERANCE_CONDITION: f32 = 1e-6;

// TODO: Consider making this densely packed
pub struct LU {
    // Stores both lower and upper in the matrix
    // Lower[i,i] := 1;
    pub matrix: NdArray,
}


impl LU {
    pub fn left_apply_l(&self, target:&mut NdArray) {
        // LA = Output
        debug_assert_eq!(target.dims[0], self.matrix.dims[1]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in (0..rows).rev() {
            for j in 0..tcols {
                for k in 0..i {
                    target.data[ i * tcols + j] += self.matrix.data[ i * cols + k] * target.data[ k * tcols + j];
                }
            }
        }
    }
    pub fn left_apply_u(&self, target:&mut NdArray) {
        // UA = Output
        debug_assert_eq!(target.dims[0], self.matrix.dims[1]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in 0..rows {
            for j in 0..tcols {
                target.data[ i * tcols + j] *= self.matrix.data[i * cols + i];
                for k in i+1..cols {
                    target.data[ i * tcols + j] += self.matrix.data[ i * cols + k] * target.data[ k * cols + j ]; 
                }
            }
        }
    }
    pub fn right_apply_l(&self, target:&mut NdArray) {
        // AL = Output
        debug_assert_eq!(target.dims[1], self.matrix.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in 0..trows {
            for j in 0..rows {
                for k in i+1..rows {
                    target.data[ i * tcols + j] += target.data[ i * tcols + k ] * self.matrix.data[ k * cols + j ];
                }
            }
        }
    }
    pub fn right_apply_u(&self, target:&mut NdArray) {
        // AU = Output
        debug_assert_eq!(target.dims[1], self.matrix.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in 0..trows {
            for j in (0..rows).rev() {
                target.data[i * tcols + j] *= self.matrix.data[j * cols + j];
                for k in 0..j {
                    target.data[i * tcols + j] += target.data[i * tcols + k] * self.matrix.data[k * cols + j];
                }
            }
        }
    }
    pub fn reconstruct(&self) -> NdArray {
        debug_assert_eq!(self.matrix.dims[0], self.matrix.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let dims = self.matrix.dims.clone();
        let mut data = vec![0_f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..=i.min(j) {
                    if k == i { data[ i * cols + j] += self.matrix.data[ i * cols + j]; }
                    else { data[i * cols + j] += self.matrix.data[ i * cols + k ] * self.matrix.data[ k * cols + j]; }
                }
            }
        }
        NdArray { dims, data }
    }
}


pub fn lu_decompose(mut matrix: NdArray) -> LU {
    // A[j, *] = c *A[i, *]
    // => c = A[i,j] / A[j,j]
    // could be extended to non-square matrices
    debug_assert_eq!(matrix.dims[0], matrix.dims[1]);
    let (rows, cols) = (matrix.dims[0], matrix.dims[1]);

    for i in 0..rows {
        for j in 0..cols {
            for k in 0..i {
                matrix.data[i * cols + j] -= matrix.data[i * cols + k] * matrix.data[k * cols + j]
            }
            if i > j {
                // NOTE: Can store this directly if we want to merge but need virtual methods
                matrix.data[i * cols + j] = matrix.data[i * cols + j] / matrix.data[j * cols + j]
            } else {
                matrix.data[i * cols + j] = matrix.data[i * cols + j] / matrix.data[i * cols + i]
            }
        }
    }
    LU { matrix }
}
