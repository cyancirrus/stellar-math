use crate::algebra::ndmethods::{create_identity_matrix, create_identity_rectangle};
use crate::algebra::vector::dot_product;
use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;

#[derive(Debug)]
pub struct QrDecomposition {
    // Qn..Q1 * A = R;
    // A = (Qn..Q1)'R
    // A = Q1'Q2'..QnR
    // A = Q1Q2..QnR
    // Q := product Q1..Qn
    pub rows: usize, // rows in input matrix
    pub cols: usize, // cols in input matrix
    pub card: usize, // count householder transforms
    pub projections: Vec<HouseholderReflection>,
    pub triangle: NdArray,
}

impl QrDecomposition {
    pub fn new(mut x: NdArray) -> Self {
        let (rows, cols) = (x.dims[0], x.dims[1]);
        let card = rows.min(cols) - (rows <= cols) as usize;
        let mut projections = Vec::with_capacity(card);
        let mut w = vec![0_f32; rows];
        for o in 0..card {
            let column_vector = (o..rows)
                .map(|r| x.data[r * cols + o])
                .collect::<Vec<f32>>();
            let proj = householder_params(column_vector);
            // x'A
            for j in o..cols {
                for i in o..rows {
                    w[j] += proj.vector[i - o] * x.data[i * cols + j];
                }
                w[j] *= proj.beta;
                for i in o..rows {
                    x.data[i * cols + j] -= proj.vector[i - o] * w[j];
                }
                w[j] = 0_f32;
            }
            projections.push(proj);
        }
        // A ~ M[m,n]
        // QR(A) -> Q ~ M[m,n], R ~ M[n,n];
        x.data.truncate(cols * cols);
        x.dims[0] = cols;
        for i in 1..cols {
            // for j in 0..i.min(cols) {
            for j in 0..i {
                x.data[i * cols + j] = 0_f32
            }
        }
        // If wanted positive elements for thetriangular matrix diagonal
        // for i in 0..card {
        //     if x.data[i*cols + i] < 0_f32 {
        //         for e in &mut projections[i].vector {
        //             *e = - *e;
        //         }
        //     }
        // }
        Self {
            rows,
            cols,
            card,
            projections,
            triangle: x,
        }
    }

    pub fn projection_matrix(&self) -> NdArray {
        // Iteration is decreasing due to constraints
        // Computes H[i] <- f(Householder i, Hi-1)
        // I - Buu'
        // H[i] * H[i-1] = H[i] - B[i-1](H[i]u[i-1])u'[i-1]
        // Hu := w
        // H[i+1] -= B[i-1] *w[i]u'[i-1]

        let card = self.card;
        let mut matrix = create_identity_rectangle(self.rows, self.cols);
        let mut w: Vec<f32> = vec![0_f32; self.rows];
        // Justification for using rows as column when we are using column major form
        // A ~ Matrix[i, j]
        // QR(A) -> (Q, R)
        // Q ~ M[i, i]
        for p in (0..card).rev() {
            let proj = &self.projections[p];
            for i in p..self.rows {
                for j in p..self.cols {
                    w[i] += matrix.data[i * self.cols + j] * proj.vector[j - p];
                }
                w[i] *= proj.beta;
                for j in p..self.cols {
                    matrix.data[i * self.cols + j] -= w[i] * proj.vector[j - p];
                }
                w[i] = 0_f32;
            }
        }
        matrix
    }
    pub fn triangle_rotation(&mut self) {
        // Specifically for the Schur algorithm which requires square matrices
        // A' = Q'AQ = Q'(QR)Q = RQ
        debug_assert!(self.rows == self.cols);
        let mut sum;
        for p in 0..self.card {
            let proj = &self.projections[p];
            for i in p..self.rows {
                sum = 0f32;
                for j in p..self.cols {
                    sum += self.triangle.data[i * self.cols + j] * proj.vector[j - p];
                }
                sum *= proj.beta;
                for j in p..self.cols {
                    self.triangle.data[i * self.cols + j] -= sum * proj.vector[j - p];
                }
            }
        }
    }
    pub fn left_apply_qt(&self, target: &mut NdArray) {
        // f(X) :: Q'X
        // debug_assert!(target.dims[0] == self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        let mut sum;
        for p in 0..self.card {
            let proj = &self.projections[p];
            // ( I - Bvv') is symmetric order matters
            for j in 0..tcols {
                sum = 0f32;
                for i in p..trows.min(self.rows) {
                    sum += proj.vector[i - p] * target.data[i * tcols + j];
                }
                sum *= proj.beta;
                for i in p..trows.min(self.rows) {
                    target.data[i * tcols + j] -= sum * proj.vector[i - p];
                }
            }
        }
        target.data.truncate(self.cols * tcols);
        target.dims[0] = self.cols;
    }
    pub fn left_apply_q(&self, target: &mut NdArray) {
        // f(X) :: QX
        // H[i]*X = X - Buu'X
        // w = u'X
        // debug_assert!(target.dims[0] == self.cols);
        let mut sum;
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for p in (0..self.card).rev() {
            let proj = &self.projections[p];
            for j in 0..tcols {
                sum = 0f32;
                for i in p..trows.min(self.rows) {
                    sum += proj.vector[i - p] * target.data[i * tcols + j];
                }
                sum *= proj.beta;
                for i in p..trows.min(self.rows) {
                    target.data[i * tcols + j] -= sum * proj.vector[i - p];
                }
            }
        }
        target.data.truncate(self.rows * tcols);
        target.dims[0] = self.rows;
    }
    pub fn right_apply_q(&self, target: &mut NdArray) {
        // f(X) :: XQ
        // H[i]*X = X - Buu'X
        // debug_assert!(target.dims[0] == self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        let mut sum;
        for p in 0..self.card {
            let proj = &self.projections[p];
            for i in 0..trows {
                sum = 0f32;
                // inner product of a[i][*] and u[p]
                for j in p..tcols.min(self.cols) {
                    sum += target.data[i * tcols + j] * proj.vector[j - p];
                }
                sum *= proj.beta;
                for j in p..tcols.min(self.cols) {
                    target.data[i * tcols + j] -= sum * proj.vector[j - p];
                }
            }
        }
        target.data.truncate(trows * self.cols);
        target.dims[1] = self.cols;
    }
    pub fn right_apply_qt(&self, target: &mut NdArray) {
        // f(X) :: XQ'
        // H[i]*X = X - Buu'X
        // debug_assert!(target.dims[0] == self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        let mut sum;
        for p in (0..self.card).rev() {
            let proj = &self.projections[p];
            for i in 0..trows {
                sum = 0f32;
                // inner product of a[i][*] and u[p]
                // for j in p..tcols.min(self.rows) {
                for j in p..tcols {
                    sum += target.data[i * tcols + j] * proj.vector[j - p];
                }
                // for j in p..tcols.min(self.rows) {
                for j in p..tcols {
                    target.data[i * tcols + j] -= sum * proj.beta * proj.vector[j - p];
                }
            }
        }
        target.data.truncate(trows * self.rows);
        target.dims[1] = self.rows;
    }
    fn multiply_vector(&self, mut data: Vec<f32>) -> Vec<f32> {
        // A ~ M[i,j] => Q ~ M[i,i]
        debug_assert!(data.len() == self.rows);
        // H[i+1]x = (I - buu')x  = x - b*u*(u'x)
        for p in (0..self.card).rev() {
            let mut scalar = 0_f32;
            let proj = &self.projections[p];
            for i in p..self.rows {
                scalar += data[i] * proj.vector[i - p];
            }
            for i in p..self.rows {
                data[i] -= scalar * proj.beta * proj.vector[i - p];
            }
        }
        data
    }
}
