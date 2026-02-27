use crate::structure::ndarray::NdArray;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub fn generate_zero_matrix(m: usize, n: usize) -> NdArray {
    NdArray {
        dims: vec![m, n],
        data: vec![0.0_f32; m * n],
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
    let mut data = vec![0.0_f32; n * n];
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
    let mut data = vec![0.0_f32; n];
    for i in 0..n {
        data[i] = rng.sample(StandardNormal);
    }
    data
}
