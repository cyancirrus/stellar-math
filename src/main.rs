#![allow(unused_variables, dead_code, unused_imports, unused_mut)]
use std::hint::black_box;
use std::time::Instant;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::algebra::ndmethods::{create_identity_matrix, matrix_mult, transpose};
use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::qr::QrDecomposition;
use stellar::decomposition::svd::golub_kahan;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};
use stellar::solver::randomized_svd::{RandomizedSvd, RankKSvd};
use stellar::structure::ndarray::NdArray;
const CONVERGENCE_CONDITION: f32 = 1e-4;

struct HouseholderMatrix {
    projs: Vec<f32>, // 2d storage
    betas: Vec<f32>,
}

/// QrDecomp
///
/// * h: HouseholderMatrix - row major form
/// * t: Upper triangular Ndarray - row major form
/// * buffer: non-cleared buffer for reuse
/// * rows: number of rows in the original matrix A
/// * cols: number of cols in the original matrix A
/// * card: rows.min(j) number of household transforms
pub struct QrDecomp {
    h: HouseholderMatrix,
    t: NdArray,
    rows: usize,
    cols: usize,
    card: usize,
}

impl HouseholderMatrix {
    pub fn new(rows:usize, card: usize) -> Self {
        Self {
            projs: vec![0_f32; rows * (card-1)],
            betas: vec![0_f32; card-1],
        }
    }
    fn params(&mut self, u: &[f32], card:usize, p: usize) -> (&[f32], &f32) {
        let row_offset = card * p;
        let n = card - p;
        let v = &mut self.projs[row_offset..row_offset + n];
        let beta = &mut self.betas[p];
        v.copy_from_slice(u);
        let mut max_element = f32::NEG_INFINITY;
        for val in v.iter() {
            // max_element = max_element.max(val.abs());
            max_element = max_element.max(*val);
        }
        let mut magnitude_squared = 0f32;
        for j in v.into_iter() {
            *j /= max_element;
            magnitude_squared += *j * *j;
        }
        let norm = magnitude_squared.sqrt();
        let sign = v[0].signum();
        let tmp = v[0];
        v[0] += sign * norm;
        magnitude_squared += 2f32 * sign * tmp * norm + magnitude_squared;
        *beta = 2f32 / magnitude_squared;
        (v, beta)
    }
}

impl QrDecomp{
    pub fn new(mut t: NdArray) -> Self {
        let (rows, cols) = (t.dims[0], t.dims[1]);
        let card = rows.min(cols);
        let mut h= HouseholderMatrix::new(rows, card);
        let mut buffer = vec![0f32; rows.max(cols)];
        for p in 0..card-1 {
            for i in p..rows {
                buffer[i] = t.data[ i * cols + p];
            }
            let (proj, beta) = h.params(&buffer[p..rows], card, p);
            // w' = u'T
            buffer.fill(0f32);
            for i in p..card {
                let row_offset = i * cols;
                for j in 0..cols {
                    buffer[j] += proj[i - p] * t.data[ row_offset + j];
                }
            }
            // T -= B uw'
            for i in p..card {
                let scalar = beta * proj[i - p];
                let row_offset = i * cols;
                for j in 0..cols {
                    t.data[ row_offset + j] -= scalar * buffer[j];
                }
            }
        }
        for i in 1..rows {
            for j in 0..i {
                t.data[i * cols + j] = 0_f32
            }
        }
        Self {
            h,
            t,
            rows,
            cols,
            card,
        }
    }
}


// Find a way to outer apply the matrix to make this j on the inside
impl QrDecomp {
    pub fn left_apply_q(&self, target: &mut NdArray) {
        // f(X) :: QX
        // H[i]*X = X - Buu'X
        // w = u'X
        debug_assert!(target.dims[0] == self.cols);
        let tcols = target.dims[1];
        let mut buffer = vec![0f32;tcols];
        for p in (0..self.card - 1).rev() {
            let proj = &self.h.projs[self.card * p .. self.card * (p + 1)];
            let beta = self.h.betas[p];
            // w' = u'X
            for i in p..self.rows {
                let row_offset = i * tcols;
                for j in 0..tcols {
                    buffer[j] += proj[i-p] * target.data[ row_offset + j];
                }
            }
            // X -= B uw'
            for i in p..self.rows {
                let scalar = beta * proj[i - p];
                let row_offset = i * tcols;
                for j in 0..tcols {
                    target.data[ row_offset + j] -= scalar * buffer[j];
                }
            }
            buffer.fill(0f32);
        }
        target.resize_rows(self.card);
    }
    pub fn left_apply_qt(&self, target: &mut NdArray) {
        // f(X) :: QX
        // H[i]*X = X - Buu'X
        // w = u'X
        // debug_assert!(target.dims[0] == self.cols);
        debug_assert!(target.dims[0] == self.cols);
        let tcols = target.dims[1];
        let mut buffer = vec![0f32;tcols];
        for p in 0..self.card - 1 {
            let proj = &self.h.projs[self.card * p .. self.card * (p + 1)];
            let beta = self.h.betas[p];
            // w' = u'X
            for i in p..self.rows {
                let row_offset = i * tcols;
                for j in 0..tcols {
                    buffer[j] += proj[i-p] * target.data[ row_offset + j];
                }
            }
            // X -= B uw'
            for i in p..self.rows {
                let scalar = beta * proj[i - p];
                let row_offset = i * tcols;
                for j in 0..tcols {
                    target.data[ row_offset + j] -= scalar * buffer[j];
                }
            }
            buffer.fill(0f32);
        }
        target.resize_rows(self.card);
    }
    pub fn right_apply_q(&self, target: &mut NdArray) {
        // f(X) :: XQ
        // H[i]*X = X - Buu'X
        // debug_assert!(target.dims[0] == self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        let mut sum;
        for p in 0..self.card-1 {
            let proj = &self.h.projs[self.card * p..self.card * (p + 1)];
            let beta = self.h.betas[p];
            for i in 0..trows {
                sum = 0f32;
                // inner product of a[i][*] and u[p]
                for j in p..tcols.min(self.card) {
                    sum += target.data[i * tcols + j] * proj[j - p];
                }
                sum *= beta;
                for j in p..tcols.min(self.card) {
                    target.data[i * tcols + j] -= sum * proj[j - p];
                }
            }
        }
        target.resize_cols(self.card);
    }
    pub fn right_apply_qt(&self, target: &mut NdArray) {
        // f(X) :: XQ'
        // H[i]*X = X - Buu'X
        // debug_assert!(target.dims[0] == self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        let mut sum;
        for p in (0..self.card-1).rev() {
            let proj = &self.h.projs[self.card * p..self.card * (p + 1)];
            let beta = self.h.betas[p];
            for i in 0..trows {
                sum = 0f32;
                // inner product of a[i][*] and u[p]
                for j in p..tcols.min(self.card) {
                    sum += target.data[i * tcols + j] * proj[j - p];
                }
                sum *= beta;
                for j in p..tcols.min(self.card) {
                    target.data[i * tcols + j] -= sum * proj[j - p];
                }
            }
        }
        target.resize_cols(self.card);
    }
}

fn check_householder_matrix() {
    let n = 5;
    let x = generate_random_matrix(n, n);
    println!("X {x:?}");
    let qr_old = QrDecomposition::new(x.clone());
    let qr_new = QrDecomp::new(x.clone());
    let y = generate_random_matrix(n, n);
    // let y = x.transpose();
    let mut y_expect= y.clone();
    let mut y_actual = y.clone();
    qr_old.left_apply_qt(&mut y_expect);
    qr_new.left_apply_qt(&mut y_actual);

    println!("Y_expected {y_expect:?}");
    println!("Y_actual {y_actual:?}");

}
fn main() {
    check_householder_matrix();
}
