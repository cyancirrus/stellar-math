#![allow(dead_code, unused_imports, unused_mut, unused_variables)]
use std::time::Instant;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::qr::QrDecomposition;
use stellar::decomposition::svd::golub_kahan;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};
use stellar::solver::randomized_svd::{RandomizedSvd, RankKSvd};
use stellar::structure::ndarray::NdArray;
use stellar::algebra::ndmethods::{create_identity_matrix, matrix_mult, transpose};
use std::hint::black_box;
const CONVERGENCE_CONDITION: f32 = 1e-4;

struct HouseholderMatrix {
    card: usize,
    projs: Vec<f32>, // 2d storage
    betas: Vec<f32>,
}
impl HouseholderMatrix {
    fn new(card:usize) -> Self {
        Self {
            card,
            projs:vec![0_f32; card * card + card],
            betas:vec![0_f32; card],
        }
    }
    fn householder_params(&mut self, u: &[f32], k:usize) {
        let row_offset = self.card * k;
        let n = self.card - k;
        let v = &mut self.projs[row_offset..row_offset + n];
        v.copy_from_slice(u);
        let mut max_element = 0f32;
        for val in v.iter() {
            max_element = max_element.max(val.abs());
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
        self.betas[k] = 2f32 / magnitude_squared;
    }
}

// Find a way to outer apply the matrix to make this j on the inside
impl HouseholderMatrix {
    pub fn apply_q_left(&self, target: &mut NdArray) {
        // f(X) :: QX
        // H[i]*X = X - Buu'X
        // w = u'X
        // debug_assert!(target.dims[0] == self.cols);
        let rows = self.card;
        let mut sum;
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for p in (0..self.card-1).rev() {
            let proj = &self.projs[self.card * p..self.card * (p + 1)];
            let beta = self.betas[p];
            for j in 0..tcols {
                sum = 0f32;
                for i in p..trows.min(self.card) {
                    sum += proj[i - p] * target.data[i * tcols + j];
                }
                sum *= beta;
                for i in p..trows.min(self.card) {
                    target.data[i * tcols + j] -= sum * proj[i - p];
                }
            }
        }
        target.data.truncate(self.card * tcols);
        target.dims[0] = self.card;
    }
}



fn main() {
}
