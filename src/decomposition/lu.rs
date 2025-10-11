use crate::algebra::ndmethods::create_identity_matrix;
use crate::structure::ndarray::NdArray;

const TOLERANCE_CONDITION: f32 = 1e-6;

// NOTE: current implementation doesn't handle pivoting

pub struct LuDecomposition {
    // matrix contains both lower and upper logically
    // lower l[i,i] := 1; // lower and upper share diagonal
    pub matrix: NdArray,
}

impl LuDecomposition {
    pub fn find_determinant(&self) -> f32 {
        // |A B| = |A| |B|
        // = diag |A| diag |B|
        // = diag |B|
        let n = self.matrix.dims[0];
        let mut det = 1_f32;
        for i in 0..n {
            det *= self.matrix.data[i * n + i];
        }
        det
    }
    // for matrices
    pub fn left_apply_l(&self, target: &mut NdArray) {
        // LA = Output
        debug_assert_eq!(target.dims[0], self.matrix.dims[1]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in (1..rows).rev() {
            for j in 0..tcols {
                // lii == 1
                for k in 0..i {
                    target.data[i * tcols + j] +=
                        self.matrix.data[i * cols + k] * target.data[k * tcols + j];
                }
            }
        }
    }
    pub fn left_apply_u(&self, target: &mut NdArray) {
        // UA = Output
        debug_assert_eq!(target.dims[0], self.matrix.dims[1]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in 0..rows {
            for j in 0..tcols {
                target.data[i * tcols + j] *= self.matrix.data[i * cols + i];
                for k in i + 1..cols {
                    target.data[i * tcols + j] +=
                        self.matrix.data[i * cols + k] * target.data[k * tcols + j];
                }
            }
        }
    }
    pub fn right_apply_l(&self, target: &mut NdArray) {
        // AL = Output
        debug_assert_eq!(target.dims[1], self.matrix.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in 0..trows {
            for j in 0..rows {
                for k in j + 1..rows {
                    target.data[i * tcols + j] +=
                        target.data[i * tcols + k] * self.matrix.data[k * cols + j];
                }
            }
        }
    }
    pub fn right_apply_u(&self, target: &mut NdArray) {
        // AU = Output
        debug_assert_eq!(target.dims[1], self.matrix.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in 0..trows {
            for j in (0..rows).rev() {
                target.data[i * tcols + j] *= self.matrix.data[j * cols + j];
                for k in 0..j {
                    target.data[i * tcols + j] +=
                        target.data[i * tcols + k] * self.matrix.data[k * cols + j];
                }
            }
        }
    }
    pub fn reconstruct(&self) -> NdArray {
        debug_assert_eq!(self.matrix.dims[0], self.matrix.dims[1]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let dims = self.matrix.dims.clone();
        let mut data = vec![0_f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..=i.min(j) {
                    if k == i {
                        data[i * cols + j] += self.matrix.data[i * cols + j]
                    } else {
                        data[i * cols + j] +=
                            self.matrix.data[i * cols + k] * self.matrix.data[k * cols + j]
                    }
                }
            }
        }
        NdArray { dims, data }
    }
}

impl LuDecomposition {
    pub fn left_apply_l_vec(&self, target: &mut [f32]) {
        // Lx
        debug_assert_eq!(self.matrix.dims[1], target.len());
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        for i in (0..rows).rev() {
            for k in 0..i {
                target[i] += self.matrix.data[i * cols + k] * target[k];
            }
        }
    }
    pub fn left_apply_u_vec(&self, target: &mut [f32]) {
        // Ux
        debug_assert_eq!(self.matrix.dims[1], target.len());
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        for i in 0..rows {
            target[i] *= self.matrix.data[i * cols + i];
            for k in i + 1..cols {
                target[i] += self.matrix.data[i * cols + k] * target[k];
            }
        }
    }
    pub fn right_apply_l_vec(&self, target: &mut [f32]) {
        //x'L
        debug_assert_eq!(self.matrix.dims[1], target.len());
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        for j in 0..cols {
            for k in j + 1..rows {
                target[j] += target[k] * self.matrix.data[k * cols + j]
            }
        }
    }
    pub fn right_apply_u_vec(&self, target: &mut [f32]) {
        //x'U
        debug_assert_eq!(self.matrix.dims[1], target.len());
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        for j in (0..cols).rev() {
            target[j] *= self.matrix.data[j * cols + j];
            for k in 0..j {
                target[j] += target[k] * self.matrix.data[k * cols + j]
            }
        }
    }
}

impl LuDecomposition {
    // Ax = y;
    // LuDecomposition*x = y;
    // Lz = y -> z;
    // Ux = z -> x;
    // => x

    pub fn solve_inplace(&self, y: &mut NdArray) {
        self.forward_solve_inplace(y);
        self.backward_solve_inplace(y);
    }
    pub fn solve_inplace_vec(&self, y: &mut [f32]) {
        self.forward_solve_inplace_vec(y);
        self.backward_solve_inplace_vec(y);
    }
    pub fn forward_solve_inplace(&self, y: &mut NdArray) {
        // transforms y -> z
        debug_assert_eq!(self.matrix.dims[1], y.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (y.dims[0], y.dims[1]);
        for j in 0..tcols {
            for i in 0..rows {
                for k in 0..i {
                    y.data[i * tcols + j] -= self.matrix.data[i * cols + k] * y.data[k * tcols + j];
                }
            }
        }
    }
    pub fn backward_solve_inplace(&self, z: &mut NdArray) {
        // transforms y -> z
        debug_assert_eq!(self.matrix.dims[1], z.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (z.dims[0], z.dims[1]);
        for i in (0..rows).rev() {
            for j in 0..tcols {
                for k in i + 1..cols {
                    z.data[i * tcols + j] -= self.matrix.data[i * cols + k] * z.data[k * tcols + j];
                }
                z.data[i * cols + j] /= self.matrix.data[i * cols + i];
            }
        }
    }
    pub fn forward_solve_inplace_vec(&self, y: &mut [f32]) {
        // transforms y -> z
        debug_assert_eq!(self.matrix.dims[1], y.len());
        let cols = self.matrix.dims[1];
        for i in 0..cols {
            for k in 0..i {
                y[i] -= self.matrix.data[i * cols + k] * y[k]
            }
        }
    }
    pub fn backward_solve_inplace_vec(&self, z: &mut [f32]) {
        // transforms z -> x
        debug_assert_eq!(self.matrix.dims[1], z.len());
        let cols = self.matrix.dims[1];
        for i in (0..cols).rev() {
            for k in i + 1..cols {
                z[i] -= self.matrix.data[i * cols + k] * z[k]
            }
            z[i] /= self.matrix.data[i * cols + i]
        }
    }
}

pub fn lu_decompose(mut matrix: NdArray) -> LuDecomposition {
    // A[j, *] = c *A[i, *]
    // => c = A[i,j] / A[j,j]
    debug_assert_eq!(matrix.dims[0], matrix.dims[1]);
    let (rows, cols) = (matrix.dims[0], matrix.dims[1]);

    // for lij, we need knowledge of ujj due to formula
    // this means that we need u[0..i.min(j)] upper triangular calculated
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..i.min(j) {
                matrix.data[i * cols + j] -= matrix.data[i * cols + k] * matrix.data[k * cols + j]
            }
            if i > j {
                matrix.data[i * cols + j] /= matrix.data[j * cols + j];
            }
        }
    }
    LuDecomposition { matrix }
}
