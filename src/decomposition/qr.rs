use crate::algebra::ndmethods::create_identity_matrix;
use crate::algebra::vector::dot_product;
use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;

#[derive(Debug)]
pub struct QrDecomposition {
    pub rows: usize, // rows in input matrix
    pub cols: usize, // cols in input matrix
    pub card: usize, // count householder transforms 
    pub projections: Vec<HouseholderReflection>,
    pub triangle: NdArray,
}

pub fn qr_decompose(mut x: NdArray) -> QrDecomposition {
    let (rows, cols) = (x.dims[0], x.dims[1]);
    let card = rows.min(cols) - (rows<=cols) as usize;
    let mut projections = Vec::with_capacity(card);
    let mut w = vec![0_f32; rows];
    for o in 0..card {
        let column_vector = (o..rows)
            .into_par_iter()
            .map(|r| x.data[r * cols + o])
            .collect::<Vec<f32>>();
        let proj = householder_params(column_vector);
        for j in o..cols {
            for i in o..rows {
                w[j] += proj.vector[i - o] * x.data[i * cols + j];
            }
            w[j] *= proj.beta;
            // tanspose vector
            for i in o..rows {
                x.data[i * cols + j] -= proj.vector[i - o] * w[j];
            }
            w[j] = 0_f32;
        }
        projections.push(proj);
    }
    for i in 1..rows {
        for j in 0..i {
            x.data[i * cols + j] = 0_f32
        }
    }
    QrDecomposition::new(rows, cols, card, projections, x)
}

impl QrDecomposition {
    pub fn new(
        rows: usize,
        cols: usize,
        card: usize,
        projections: Vec<HouseholderReflection>,
        triangle: NdArray,
    ) -> Self {
        Self {
            rows,
            cols,
            card,
            projections,
            triangle,
        }
    }
    pub fn projection_matrix(&self) -> NdArray {
        // I - Buu'
        // H[i+1] * H[i] = H[i+1] - B[i](H[i+1]u[i])u'[i]
        // Hu := w
        // H[i+1] -= B[i] *w[i+1]u'[i]

        let card = self.card;
        let mut matrix = create_identity_matrix(self.rows);
        let mut w: Vec<f32> = vec![0_f32; self.rows];
        // A ~ Matrix[i, j]
        // QR(A) -> (Q, R)
        // Q ~ M[i, i]
        // this is why self.rows is used as column indexing
        for p in (0..card).rev() {
            let proj = &self.projections[p];
            for i in p..self.rows {
                for j in p..self.cols {
                    w[i] += matrix.data[i * self.rows + j] * proj.vector[j - p];
                }
                w[i] *= proj.beta;
                for j in p..self.cols {
                    matrix.data[i * self.rows + j] -= w[i] * proj.vector[j - p];
                }
                w[i] = 0_f32;
            }
        }
        matrix
    }
    pub fn triangle_rotation(&mut self) {
        // Specifically for the Schur algorithm which requires square matrices
        // A' = Q'AQ = Q'(QR)Q = RQ
        debug_assert!(self.rows==self.cols);
        let mut w: Vec<f32> = vec![0_f32; self.rows];
        for p in 0..self.card {
            let proj = &self.projections[p];
            for i in p..self.rows {
                for j in p..self.cols {
                    w[i] += self.triangle.data[i * self.cols + j] * proj.vector[j - p];
                }
                w[i] *= proj.beta;
                for j in p..self.cols {
                    self.triangle.data[i * self.cols + j] -= w[i] * proj.vector[j - p];
                }
                w[i] = 0_f32;
            }
        }
    }
    pub fn left_multiply(&self, target: &mut NdArray) {
        // AX -> QX
        // H[i]*X = X - Buu'X
        // w = u'X
        debug_assert!(target.dims[0] == self.rows);
        let mut w = vec![0_f32; self.rows];
        for p in (0..self.card).rev() {
            let proj = &self.projections[p];
            for j in 0..self.cols {
                for i in p..self.rows {
                    w[j] += proj.vector[i - p] * target.data[i * self.cols + j];
                }
                for i in p..self.rows {
                    target.data[i * self.cols + j] -= proj.beta * w[j] * proj.vector[i - p];
                }
                w[j] = 0_f32;
            }
        }
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
