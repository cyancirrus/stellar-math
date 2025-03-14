#![allow(warnings)]
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
use std::cmp::min;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn identity(x: f32) -> f32 {
    x
}

pub fn loss_squared(prediction: Vec<f32>, result: Vec<f32>) -> f32 {
    let loss = prediction
        .par_iter()
        .zip(result.par_iter())
        .map(|(p, r)| (p - r) * (p - r))
        .sum();
    loss
}

pub fn cross_apply(x: &[f32], y: &[f32], f_enum: fn(usize, f32, usize, f32) -> f32) -> Vec<Vec<f32>> {
    let rows = x.len();
    let cols = y.len();
    let mut matrix = vec![vec![0_f32; cols]; rows];

    for row in 0..rows {
        for col in 0..cols {
            matrix[row][col] = f_enum(row, x[row], col, y[col]);
        }
    }
    matrix
}

pub fn cross_product(x: Vec<f32>, y: Vec<f32>) -> Vec<Vec<f32>> {
    fn product(_: usize, x: f32, _: usize, y: f32) -> f32 {
        x * y
    }
    cross_apply(&x, &y, product)
}

pub fn outer_product(x: Vec<f32>) -> Vec<f32> {
    // returns a a symetric matrix of length x length
    let length = x.len();
    assert!(length > 0, "needs to have non-zero length");
    let mut new_data = vec![0_f32; length * length];
    for i in 0..length {
        for j in 0..length {
            new_data[i * length + j] = x[i] * x[j];
        }
    }
    new_data
}
