use crate::algebra::vector::{dot_product, magnitude, vector_multiply, vector_diff};
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;

#[derive(Debug)]
pub struct HouseholderReflection {
    pub beta: f32,        // store 2 / u'u
    pub vector: Vec<f32>, // stores reflection u
}

impl HouseholderReflection {
    pub fn new(beta: f32, vector: Vec<f32>) -> Self {
        Self { beta, vector }
    }
}

const EPSILON: f32 = 1e-6;

pub fn householder_params(mut u: Vec<f32>) -> HouseholderReflection {
    let length = u.len();
    let mut max_element = f32::NEG_INFINITY;
    let mut magnitude_squared = 0_f32;
    for i in 0..length {
        max_element = max_element.max(u[i]);
    }
    if max_element.abs() < EPSILON {
        return HouseholderReflection::new(0_f32, vec![0_f32]);
    }
    for i in 0..length {
        u[i] /= max_element;
        magnitude_squared += u[i].powi(2);
    }
    let sign = u[0].signum();
    let tmp = u[0];
    u[0] += sign * magnitude_squared.sqrt();
    magnitude_squared += 2_f32 * sign * tmp * magnitude_squared.sqrt() + magnitude_squared;
    HouseholderReflection::new(2_f32 / magnitude_squared, u)
}
//pub fn householder_params(x: &[f32]) -> HouseholderReflection {
//    //TODO: change the above to take ownership of the vector
//    // already cloning it and QR requires column and golub kahn requires rows

//    // pub fn householder_params(mut x: Vec<f32>) -> HouseholderReflection {
//    let length = x.len();
//    let mut max_element = f32::NEG_INFINITY;
//    let mut magnitude_squared = 0_f32;
//    for i in 0..length {
//        max_element = max_element.max(x[i]);
//    }
//    if max_element.abs() < EPSILON {
//        return HouseholderReflection::new(0_f32, vec![0_f32]);
//    }
//    let mut u = vec![0_f32; length];
//    for i in 0..length {
//        let result = x[i] / max_element;
//        u[i] = result;
//        magnitude_squared += result.powi(2);
//    }
//    let sign = u[0].signum();
//    let tmp = u[0];
//    u[0] += sign * magnitude_squared.sqrt();
//    magnitude_squared += 2_f32 * sign * tmp * magnitude_squared.sqrt() + magnitude_squared;
//    HouseholderReflection::new(2_f32 / magnitude_squared, u)
//}
