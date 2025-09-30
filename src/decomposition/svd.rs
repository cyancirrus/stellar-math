use crate::algebra::ndmethods::{create_identity_matrix, tensor_mult, transpose};
use crate::algebra::vector::{initialize_unit_vector, magnitude};
use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;

const EPSILON:f32 = 1e-6;

// TODO: Optimize this method using the standard multiplication
pub fn golub_kahan_explicit(mut a: NdArray) -> NdArray {
    // TODO: refactor so householder_params mutates in place
    let (rows, cols)  = (a.dims[0], a.dims[1]);
    let card = rows.min(cols) - (rows <= cols) as usize;
    let mut householder: HouseholderReflection;

    let mut new: NdArray;
    // 5, 2 // should be two
    // 2, 5 should be one
    // 3, 3 should be 2
    for o in 0..card {
        new = create_identity_matrix(rows);
        let column_vector = (o..rows)
            .into_par_iter()
            .map(|r| a.data[r * cols + o])
            .collect::<Vec<f32>>();
        householder = householder_params(column_vector);
        if householder.beta < EPSILON {
            continue;
        }
        for i in 0..rows - o {
            for j in 0..cols - o {
                new.data[(o + i) * cols + j + o] -=
                    householder.beta * householder.vector[i] * householder.vector[j]
            }

        }
        a = tensor_mult(4, &new, &a);
        // 4 
        if o < card - 1 {
            new = create_identity_matrix(cols);
            // let row_vector: Vec<f32> = a.data[(o * cols) + 1..(o + 1) * cols].to_vec();
            let row_vector: Vec<f32> = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
            householder = householder_params(row_vector);

            for i in 0..rows - o - 1 {
                for j in 0..cols - o - 1 {
                    new.data[(o + i + 1) * cols + (j + o + 1)] -=
                        householder.beta * householder.vector[i] * householder.vector[j];
                }
            }
            a = tensor_mult(4, &a, &new);
        }
    }
    a
}
