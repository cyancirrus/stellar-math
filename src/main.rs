#![allow(unused)]
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

trait DenseTriangle {
    fn l_mult(&self, matrix: &mut NdArray);
    fn r_mult(&self, matrix: &mut NdArray);
}
struct LowerTriangle {
    t: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl LowerTriangle {
    fn check(&self) {
        let mut idx = 0;
        let t = &self.t;
        for i in 0..self.rows {
            idx += i;
            for j in 0..=i {
                let jdx = idx + j;
                print!(" {:}", t[jdx]);
            }
            print!("\n");
        }
    }
    fn check_reverse(&self) {
        let mut idx = self.t.len();
        let t = &self.t;
        for i in (0..self.rows).rev() {
            idx -= (i + 1);
            for j in 0..=i {
                let jdx = idx + j;
                print!(" {:}", t[jdx]);
            }
            print!("\n");
        }
    }
    fn mult_new(&self, matrix: &NdArray) -> NdArray {
        let t = &self.t;
        let m = &matrix.data;
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (matrix.dims[0], matrix.dims[1]);
        let card = self.t.len();
        let mut o = vec![0f32; rows * tcols];
        let mut idx = 0;
        for i in 0..self.rows {
            idx += i;
            for k in 0..=i {
                let scalar = t[idx + k];
                for j in 0..tcols {
                    let rdx = k * tcols + j;
                    let tdx = i * cols + j;
                    o[tdx] += scalar * m[rdx];
                }
            }
        }
        NdArray {
            dims: vec![rows, tcols],
            data: o,
        }
    }
    fn mult_base(&self, matrix: &mut NdArray) {
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (matrix.dims[0], matrix.dims[1]);
        debug_assert_eq!(cols, trows);
        let t = &self.t;
        let m = &mut matrix.data;
        let mut idx = self.t.len();
        let mut tmp;
        let mut rdx;
        let mut tdx = trows * tcols;
        for i in (0..self.rows).rev() {
            idx -= (i + 1);
            tdx -= tcols;
            let scalar = t[idx + i];
            for j in 0..tcols {
                m[i * tcols + j] *= scalar;
            }
            for j in 0..tcols {
                tmp = 0f32;
                rdx = j;
                for k in 0..i {
                    tmp += t[idx + k] * m[rdx];
                    rdx += tcols;
                }
                m[tdx + j] += tmp;
            }
        }
    }
    fn mult(&self, matrix: &mut NdArray) {
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (matrix.dims[0], matrix.dims[1]);
        debug_assert_eq!(cols, trows);
        let t = &self.t;
        let m = &mut matrix.data;
        let mut idx = self.t.len();
        let mut tdx = trows * tcols;
        for i in (0..self.rows).rev() {
            idx -= (i + 1);
            tdx -= tcols;
            let scalar = t[idx + i];
            let tri_row = &t[idx..idx + i];
            let (tar_upper, tar_row) = m.split_at_mut(tdx);
            for t in tar_row[..tcols].iter_mut() {
                *t *= scalar;
            }
            let mut kdx = 0;
            for k in 0..i {
                let scalar = tri_row[k];
                let k_row = &tar_upper[kdx..kdx + tcols];
                for (t, k) in tar_row.iter_mut().zip(k_row.iter()) {
                    *t += scalar * *k;
                }
                kdx += tcols;
            }
        }
    }
}

impl LowerTriangle {
    // Computes T * M where T is lower triangular and packed
    fn mult_inplace(&self, matrix: &mut NdArray) {
        let (rows, cols) = (self.rows, self.cols);
        let (_, m_cols) = (matrix.dims[0], matrix.dims[1]);

        let m = &mut matrix.data;
        let t = &self.t;

        // Iterate backwards to allow in-place modification
        // without overwriting data needed for subsequent rows
        let mut t_start_idx = (rows * (rows + 1)) / 2;

        for i in (0..rows).rev() {
            let row_len = i + 1;
            t_start_idx -= row_len;

            // We need to compute: Out[i] = sum_{k=0 to i} T[i, k] * M[k]
            // Separate the diagonal T[i, i] to handle the in-place update
            let diag_val = t[t_start_idx + i];

            for j in 0..m_cols {
                let mut acc = 0.0;
                // Dot product of T[i, 0..i] and M[0..i, j]
                for k in 0..i {
                    acc += t[t_start_idx + k] * m[k * m_cols + j];
                }

                // Update: M[i, j] = T[i, i] * M[i, j] + acc
                let m_idx = i * m_cols + j;
                m[m_idx] = diag_val * m[m_idx] + acc;
            }
        }
    }
}

fn test() {
    // 6 - 3 = 3
    let t = vec![1f32, 2f32, 2f32, 3f32, 3f32, 3f32];
    let (rows, cols) = (3, 3);
    let mut tri = LowerTriangle { t, rows, cols };

    println!("top top down");
    tri.check();
    println!("bottom up");
    tri.check_reverse();
}

fn test_mult() {
    println!("********* Mutation Mut **************");
    let t = vec![4f32, 2f32, 2f32, 3f32, 3f32, 3f32];
    let (rows, cols) = (3, 3);
    let mut tri = LowerTriangle { t, rows, cols };

    let data = vec![4f32, 0f32, 0f32, 2f32, 2f32, 0f32, 3f32, 3f32, 3f32];
    let square = NdArray {
        data,
        dims: vec![rows, cols],
    };
    let input = generate_random_matrix(cols, cols);
    println!("input {input:?}");
    let expected = matrix_mult(&square, &input);
    println!("expected {expected:?}");
    let mut result = input;
    tri.mult(&mut result);
    println!("post {result:?}");
}
fn test_mult_new() {
    print!("\n");
    println!("********* Mult New **************");
    let t = vec![4f32, 2f32, 2f32, 3f32, 3f32, 3f32];
    let (rows, cols) = (3, 3);
    let mut tri = LowerTriangle { t, rows, cols };

    let data = vec![4f32, 0f32, 0f32, 2f32, 2f32, 0f32, 3f32, 3f32, 3f32];
    let square = NdArray {
        data,
        dims: vec![rows, cols],
    };
    let input = generate_random_matrix(cols, cols);
    println!("input {input:?}");
    let expected = matrix_mult(&square, &input);
    println!("expected {expected:?}");
    let result = tri.mult_new(&input);
    println!("post {result:?}");
}

fn main() {
    test();
    test_mult();
    test_mult_new();
}
