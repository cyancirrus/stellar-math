use crate::algebra::ndmethods::matrix_mult;
use crate::structure::ndarray::NdArray;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub fn generate_identity_vector(m: usize, n: usize) -> Vec<f32> {
    let mut vector = vec![0f32; m * n];
    let mut idx = 0;
    for _ in 0..m {
        vector[idx] = 1f32;
        idx += 1 + n;
    }
    vector
}
pub fn generate_approx_symmetric_vector(n: usize) -> Vec<f32> {
    let a = generate_random_matrix(n, n);
    matrix_mult(&a, &a.transpose()).data
}
pub fn generate_strict_symmetric_vector(n: usize) -> Vec<f32> {
    let mut data = generate_random_vector(n * n);
    for i in 0..n {
        for j in 0..i {
            let val = data[i * n + j];
            data[j * n + i] = val;
        }
    }
    data
}
pub fn generate_zero_matrix(m: usize, n: usize) -> NdArray {
    NdArray {
        dims: vec![m, n],
        data: vec![0f32; m * n],
    }
}

pub fn generate_random_matrix(m: usize, n: usize) -> NdArray {
    let mut rng = rand::rng();
    let mut data = vec![0f32; m * n];
    for idx in 0..m * n {
        data[idx] = rng.sample(StandardNormal);
    }
    NdArray {
        dims: vec![m, n],
        data,
    }
}
pub fn generate_random_symetric(n: usize) -> NdArray {
    let mut rng = rand::rng();
    let mut data = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..i {
            let val = rng.sample(StandardNormal);
            data[i * n + j] = val;
            data[j * n + i] = val;
        }
    }
    NdArray {
        dims: vec![n, n],
        data,
    }
}
pub fn generate_random_vector(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let mut data = vec![0f32; n];
    for i in 0..n {
        data[i] = rng.sample(StandardNormal);
    }
    data
}
