use crate::algebra::ndmethods::tensor_mult;
use crate::random::generation::generate_random_matrix;
use crate::structure::ndarray::NdArray;

pub struct LuPivotDecomp {
    n: usize,
    swaps: usize,
    pivots: Vec<usize>,
    matrix: NdArray,
}
impl LuPivotDecomp {
    pub fn new(mut matrix: NdArray) -> Self {
        // Croute
        // A[j, *] = c *A[i, *]
        // => c = A[i,j] / A[j,j]
        debug_assert_eq!(matrix.dims[0], matrix.dims[1]);
        let n = matrix.dims[0];
        let mut pivots:Vec<usize> = (0..n).collect();
        let mut swaps = 0;

        // for lij, we need knowledge of ujj due to formula
        // this means that we need u[0..i.min(j)] upper triangular calculated
        for i in 0..n {
            let mut val = matrix.data[ i * n + i].abs();
            for k in i+1..n {
                let mag = matrix.data[k * n + i].abs();
                if mag > val {
                    pivots[i] = k;
                    val = mag;
                }
            }
            if pivots[i] != i {
                swaps += 1;
                for k in 0..n {
                    matrix.data.swap(i * n + k, pivots[i] * n + k);
                }
            }
            for j in 0..n {
                for k in 0..i.min(j) {
                    matrix.data[i * n + j] -= matrix.data[i * n + k] * matrix.data[k * n + j]
                }
                if i > j {
                    matrix.data[i * n + j] /= matrix.data[j * n + j];
                }
            }
        }
        Self { n, swaps, pivots, matrix }
    }
    pub fn new_dl(mut matrix: NdArray) -> Self {
        // Doolittle
        debug_assert_eq!(matrix.dims[0], matrix.dims[1]);
        let n = matrix.dims[0];
        let mut pivots: Vec<usize> = (0..n).collect();
        let mut swaps = 0;

        let mut val;
        for k in 0..n {
            val = 0.0;
            for i in k + 1..n {
                let mag = matrix.data[i * n + k].abs();
                if mag >= val {
                    pivots[k] = i;
                    val = mag;
                }
            }
            if pivots[k] != k {
                swaps += 1;
                for j in 0..n {
                    matrix.data.swap(k * n + j, pivots[k] * n + j);
                }
            }
            for i in k + 1..n {
                matrix.data[i * n + k] /= matrix.data[k * n + k];
            }
            for i in k + 1..n {
                for j in k + 1..n {
                    matrix.data[i * n + j] -= matrix.data[i * n + k] * matrix.data[k * n + j]
                }
            }
        }
        Self { n, swaps, pivots, matrix }
    }
    pub fn reconstruct(&self) -> NdArray {
        let mut data = vec![0_f32; self.n * self.n];
        let dims = vec![self.n; 2];
        for i in 0..self.n {
            for j in 0..self.n {
                // if i > j then we hit an add at a[i][k] b/c diagonal is identity;
                for k in 0..=i.min(j) {
                    if k == i {
                        data[i * self.n + j] += self.matrix.data[i * self.n + j];
                    } else {
                        data[i * self.n + j] +=
                            self.matrix.data[i * self.n + k] * self.matrix.data[k * self.n + j];
                    }
                }
            }
        }
        for (n, &k) in self.pivots.iter().enumerate().rev() {
            if n == k {
                continue;
            }
            for j in 0..self.n {
                data.swap(n * self.n + j, k * self.n + j);
            }
        }
        NdArray { dims, data }
    }
    pub fn determinant(&self) -> f32 {
        let mut det = 1f32;
        for k in 0..self.n {
            det *= self.matrix.data[ k * self.n + k];
        }
        if self.swaps & 1 == 1 {
            det = -det
        }
        det
    }
}
impl LuPivotDecomp {
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
}

impl LuPivotDecomp {
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

impl LuPivotDecomp {
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
