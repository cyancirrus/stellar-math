use crate::algebra::ndmethods::create_identity_matrix;
use crate::algebra::vector::dot_product;
use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;

#[derive(Debug)]
pub struct QrDecomposition {
    pub rows: usize,
    pub cols: usize,
    pub card: usize,
    pub projections: Vec<HouseholderReflection>,
    pub triangle: NdArray,
}

pub fn qr_decompose(mut x: NdArray) -> QrDecomposition {
    let (rows, cols, card) = (x.dims[0], x.dims[1], x.dims[0].min(x.dims[1]));
    let mut projections = Vec::with_capacity(card.saturating_sub(1));
    let mut w = vec![0_f32; rows];
    for o in 0..card.saturating_sub(1) {
        // TODO: This is a double clone refactor so golub and schur can use it
        let column_vector = (o..rows)
            .into_par_iter()
            .map(|r| x.data[r * cols + o])
            .collect::<Vec<f32>>();
        let proj = householder_params(&column_vector);
        for j in o..cols {
            for i in o..rows {
                w[j] += proj.vector[i - o] * x.data[i * cols + j];
            }
            w[j] *= proj.beta;
            // tanspose vector
        }
        for j in o..cols {
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
        let card = self.card;
        let mut matrix = create_identity_matrix(card);
        let mut w: Vec<f32> = vec![0_f32; card];
        // I - Buu'
        // H[i+1] * H[i] = H[i+1] - B[i](H[i+1]u[i])u'[i]
        // Hu := w
        // H[i+1] -= B[i] *w[i+1]u'[i]
        for p in (0..card.saturating_sub(1)).rev() {
            let proj = &self.projections[p];
            for i in p..card {
                for j in p..card {
                    w[i] += matrix.data[i * card + j] * proj.vector[j - p];
                }
                w[i] *= proj.beta;
            }
            for i in p..card {
                for j in p..card {
                    matrix.data[i * card + j] -= w[i] * proj.vector[j - p];
                }
                w[i] = 0_f32;
            }
        }
        matrix
    }
    pub fn triangle_rotation(&mut self) {
        // Specifically for the Schur algorithm
        // A' = Q'AQ = Q'(QR)Q = RQ
        let card = self.card;
        let mut w: Vec<f32> = vec![0_f32; card];
        for p in 0..self.card.saturating_sub(1) {
            let proj = &self.projections[p];
            for i in p..card {
                for j in p..card {
                    w[i] += self.triangle.data[i * card + j] * proj.vector[j - p];
                }
                w[i] *= proj.beta;
            }
            for i in p..card {
                for j in p..card {
                    self.triangle.data[i * card + j] -= w[i] * proj.vector[j - p];
                }
                w[i] = 0_f32;
            }
        }
    }
    pub fn left_multiply(&self, target: &mut NdArray) {
        // AX -> QX
        // H[i]*X = X - Buu'X
        // w = u'X
        debug_assert!(target.dims[0] == target.dims[1]);
        debug_assert!(target.dims[0] == self.card);
        let (rows, cols, card) = (target.dims[0], target.dims[1], self.card);
        let mut w = vec![0_f32; rows];
        for p in (0..card.saturating_sub(1)).rev() {
            let proj = &self.projections[p];
            for j in 0..cols {
                for i in p..rows {
                    w[j] += proj.vector[i - p] * target.data[i * cols + j];
                }
            }
            for j in 0..cols {
                for i in p..rows {
                    target.data[i * cols + j] -= proj.beta * w[j] * proj.vector[i - p];
                }
                w[j] = 0_f32;
            }
        }
    }
    fn multiply_vector(&self, mut data: Vec<f32>) -> Vec<f32> {
        debug_assert!(data.len() == self.rows);

        // H[i+1]x = (I - buu')x  = x - b*u*(u'x)
        for p in 0..self.card.saturating_sub(1) {
            let mut scalar = 0_f32;
            let proj = &self.projections[p];
            debug_assert!(self.card == proj.vector.len() + p);
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
