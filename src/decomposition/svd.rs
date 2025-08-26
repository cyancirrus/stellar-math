use crate::algebra::ndmethods::{create_identity_matrix, tensor_mult, transpose};
use crate::algebra::vector::{initialize_unit_vector, magnitude};
use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;


pub fn golub_kahan_explicit(mut a: NdArray) -> NdArray {
    let rows = a.dims[0];
    let cols = a.dims[1];
    let mut householder: HouseholderReflection;

    let mut new: NdArray;
    // println!("Should be identity {:?}", new);

    for o in 0..cols.min(rows) - 1 {
        new = create_identity_matrix(rows);
        let column_vector = (o..rows)
            .into_par_iter()
            .map(|r| a.data[r * cols + o])
            .collect::<Vec<f32>>();
        householder = householder_params(&column_vector);
        for i in 0..rows - o {
            for j in 0..cols - o {
                new.data[(o + i) * cols + j + o] -=
                    householder.beta * householder.vector[i] * householder.vector[j]
            }
        }
        println!("Left multiplication {:?}", new);
        a = tensor_mult(4, &new, &a);
        println!("Here's what the mult looks like check 0's {:?}", a);
        if o < cols.min(rows) - 2 {
            new = create_identity_matrix(rows);
            let row_vector: Vec<f32> = a.data[(o * cols) + 1..(o + 1) * cols].to_vec();
            // println!("Row vector should be dim 3 and the top row {:?}", row_vector);
            householder = householder_params(&row_vector);

            for i in 0..rows - o - 1 {
                for j in 0..cols - o - 1 {
                    new.data[(o + i + 1) * cols + (j + o + 1)] -=
                        householder.beta * householder.vector[i] * householder.vector[j];
                }
            }
            // println!("This should be the matrix {:?}", new);
            a = tensor_mult(4, &a, &new);
            // println!("This is what a looks like {:?}", a);
        }
    }
    a
}


#[allow(non_snake_case)]
pub fn golub_kahan_lanczos(a: NdArray) -> (NdArray, NdArray, NdArray) {
    let n = a.dims[0];
    let mut U = NdArray::new(a.dims.clone(), vec![0_f32; n * n]);
    let mut V = NdArray::new(a.dims.clone(), vec![0_f32; n * n]);
    let mut beta = 0_f32;
    let mut alpha;
    let mut v = initialize_unit_vector(n);
    let mut u = vec![0_f32; n];
    for i in 0..n {
        for k in 0..n {
            V.data[i * n + k] = v[k];
        }
        u.iter_mut().for_each(|val| *val *= -beta);
        {
            for j in 0..n {
                for k in 0..n {
                    u[j] += a.data[j * n + k] * v[k];
                }
            }
        }
        alpha = magnitude(&u);
        u.iter_mut().for_each(|val| *val /= alpha);
        for j in 0..n {
            U.data[i * n + j] = u[j];
        }
        {
            v.iter_mut().for_each(|val| *val *= -alpha);
            for j in 0..n {
                for k in 0..n {
                    v[j] += a.data[k * n + j] * u[k];
                }
            }
        }
        beta = magnitude(&v);
        v.iter_mut().for_each(|val| *val /= beta);
    }
    let mut S = tensor_mult(2, &U.clone(), &a);
    S = tensor_mult(3, &S, &transpose(V.clone()));
    println!("Check U {:?}", U);
    let mut O_check = tensor_mult(2, &U, &transpose(U.clone()));
    println!("Check U is orthogonal {:?}", O_check);
    O_check = tensor_mult(2, &V, &transpose(V.clone()));
    println!("Check V is orthogonal {:?}", O_check);

    (U, S, V)
}

pub fn golub_kahan(mut a: NdArray) -> NdArray {
    let rows = a.dims[0];
    let cols = a.dims[1];
    let mut householder: HouseholderReflection;
    let mut queue = vec![0_f32; rows * cols];

    for o in 0..cols.min(rows) - 1 {
        householder = householder_params(
            // column vector
            &(o..rows)
                .into_par_iter()
                .map(|r| a.data[r * cols + o])
                .collect::<Vec<f32>>(),
        );
        for i in o..rows {
            for j in o..cols {
                for k in o..rows {
                    queue[i * cols + j] += householder.beta
                        * householder.vector[i - o]
                        * householder.vector[k - o]
                        * a.data[k * cols + j];
                }
            }
        }
        for i in o..cols {
            for j in o..rows {
                a.data[i * cols + j] -= queue[i * cols + j];
            }
        }
        queue.fill(0_f32);
        if o < cols.min(rows) - 2 {
            householder = householder_params(
                // row vector
                &a.data[(o * cols) + 1..(o + 1) * cols],
            );
            for i in 0..rows {
                for j in o + 1..cols {
                    for k in 0..rows {
                        if o < k {
                            queue[i * cols + j] += householder.beta
                                * a.data[i * cols + k]
                                * householder.vector[k - o - 1]
                                * householder.vector[j - o - 1];
                        }
                    }
                }
            }
            for i in 0..rows {
                for j in o..cols {
                    a.data[i * cols + j] -= queue[i * cols + j];
                }
            }
            queue.fill(0_f32);
        }
    }
    a
}
