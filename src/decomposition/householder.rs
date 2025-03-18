use crate::structure::ndarray::NdArray;
use crate::decomposition::qr::QrDecomposition;
use crate::algebra::vector:: { dot_product, magnitude};
use rayon::prelude::*;

pub struct HouseholderReflection {
    pub beta:f32, // store 2 / u'u
    pub vector:Vec<f32>, // stores reflection u
}

impl HouseholderReflection {
    pub fn new(beta:f32, vector:Vec<f32>) -> Self {
        Self { beta, vector }
    }
}

pub fn householder_params(x: &[f32]) -> HouseholderReflection {
    let length = x.len();
    assert!(length > 0, "needs to have non-zero length");
    let dims = vec![length; 2];
    let data = vec![0_f32; length * length];
    let mut householder = NdArray::new(dims, data);
    let max_element: f32 = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut u = x.par_iter().copied()
        .map(|val| val / max_element)
        .collect::<Vec<f32>>();
    u[0] += magnitude(&u) * x[0].signum();
    let magnitude_squared = dot_product(&u, &u);
    for i in 0..length {
        householder.data[i * length + i] = 1_f32;
    }
    HouseholderReflection::new(2_f32 / magnitude_squared, u)
}

pub fn householder_factor(mut x: NdArray) -> QrDecomposition {
    let rows = x.dims[0];
    let cols = x.dims[1];
    let mut projections = Vec::with_capacity(cols.min(rows));
    
    for o in 0..cols.min(rows) {
        let column_vector = (o..rows).into_par_iter().map(|r| x.data[r*cols + o]).collect::<Vec<f32>>();
        let householder = householder_params(&column_vector);
        projections.push(householder);
        let mut queue: Vec<(usize, f32)> = vec![(0, 0_f32); (cols - o)  * (rows -o)];
        for i in 0..(rows-o).min(cols-o) {
            for j in 0..cols-o {
                // Need to compute the change for everything to the right of the initial vector
                if i <= j || j > o {
                    let sum = (0..rows-o).into_par_iter().map(|k| {
                        x.data[(k + o)*cols + (j + o)] * projections[o].beta * projections[o].vector[i] * projections[o].vector[k]
                    }).sum();
                    queue[i*(cols - o) + j].0 = (i + o)* cols + (j+ o);
                    queue[i*(cols - o) + j].1 = sum;
                }
            }
        }
        queue.iter().for_each(|q| x.data[q.0] -= q.1);
        (o+1..rows).for_each(|i| x.data[i*cols + o] = 0_f32);
    }
    QrDecomposition::new(projections, x)
}
