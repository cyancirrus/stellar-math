use crate::structure::ndarray::NdArray;
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_distr::Normal;
use rand_distr::StandardNormal;

const CONVERGENCE_CONDITION: f32 = 1e-6;

pub fn generate_random_matrix(m: usize, n: usize) -> NdArray {
    let mut rng = rand::rng();
    let mut data = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let val = rng.sample(StandardNormal);
            data[i * n + j] = val;
        }
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
