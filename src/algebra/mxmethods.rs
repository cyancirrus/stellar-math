use crate::structure::matrix::Matrix;
use rayon::prelude::*;

pub fn transpose(x: Matrix) -> Vec<f32> {
    let mut new: Vec<f32> = vec![0_f32; x.rows * x.cols];
    for i in 0..x.rows {
        for j in 0..x.cols {
            new[i * x.cols + j] = x.data[j * x.rows + i];
        }
    }
    new
}
fn transpose_optimized(x: Matrix) -> Vec<f32> {
    let length: usize = x.rows * x.cols;
    (0..length)
        .collect::<Vec<usize>>()
        .par_iter()
        .map(|i| x.data[i * x.cols % length + i / x.rows])
        .collect::<Vec<f32>>()
}

fn matrix_multiplication(left: Matrix, right: Matrix) -> Matrix {
    assert_eq!(
        left.cols, right.rows,
        "dimensions do not match in matrix mult"
    );
    let mut new: Vec<f32> = vec![0f32; left.rows * right.cols];
    let mut accum: f32 = 0f32;

    for i in 0..left.rows {
        for j in 0..right.cols {
            // common index between left and right
            for k in 0..left.cols {
                accum += left.data[i * left.rows + k] * right.data[j * right.cols + k]
            }
            new[i * left.rows + j * right.cols] = accum;
            accum = 0_f32;
        }
    }
    Matrix::new(left.rows, right.cols, new)
}
