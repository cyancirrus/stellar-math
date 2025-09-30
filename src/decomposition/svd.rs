use crate::algebra::ndmethods::{create_identity_matrix, tensor_mult, transpose, resize_rows, resize_cols};
use crate::algebra::vector::{initialize_unit_vector, magnitude};
use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;

pub fn golub_kahan_explicit(mut a: NdArray) -> NdArray {
    let (rows, cols)  = (a.dims[0], a.dims[1]);
    println!("rows {rows:}, cols {cols:}");
    let card = rows.min(cols) - (rows <= cols) as usize;
    let mut proj: HouseholderReflection;
    let mut w = vec![0f32; rows.max(cols)];
    resize_rows(4,&mut a); 
    resize_cols(4, &mut a); 
    let (rows, cols) = (4, 4);
    println!("a {a:?}");
    for o in 0..card {
        proj = householder_params(
            // column vector
            (o..rows).into_iter().map(|r| a.data[r * cols + o]).collect()
        );
        println!("here?");
        // println!("w {w:?}");
        
        // (I - bvv')A => w := bv'A
        //  A -= vw'
        for j in o..cols {
            for i in o..rows {
                w[j] += proj.vector[i - o] * a.data[ i * cols + j];
            }
            w[j] *= proj.beta;
            for i in o..rows {
                a.data[ i * cols + j] -= proj.vector[i - o] * w[j];
            }
            w[j] = 0_f32;
        }
        println!("After column {a:?}");
        // stop one early for columns because a[m,n-1] can be non-zero
        if o+1 == card  { break; }
        // [2..n]
        // [3..n]
        //
        let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
        println!("row_vector {row_vector:?}");
        proj = householder_params(
            // row vector
            a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec()
        );
        println!("proj {proj:?}");

        // A(I - bvv') => w = b * Av
        // A -= wv'
        for i in o..rows {
            for j in o+1..cols {
                w[i] += a.data[i * cols + j] * proj.vector[j - o - 1];
            }
            w[i] *= proj.beta;
        }
        println!("w {w:?}");
        for i in o..rows {
            for j in o+1..cols {
                a.data[i * cols + j] -= w[i] * proj.vector[j - o - 1];
            }
            w[i] = 0_f32;
        }
        println!("After row {a:?}");
        println!("-------------------------------");
    }
    // TODO: uncomment so flows naturally into givens
    let dims = rows.min(cols);
    resize_rows(dims,&mut a); 
    resize_cols(dims, &mut a); 
    println!("a {a:?}");
    a
}



// Aternate right optimization (column elim) and then left optimization (row elim)
// column elimination:
// (I - bvv')A => w := bv'A
//  A -= vw'
//  
// row elimination:
// A(I - bvv') => w = b * Av
// A -= wv'

// pub fn golub_kahan_explicit(mut a: NdArray) -> NdArray {
//     let (rows, cols)  = (a.dims[0], a.dims[1]);
//     println!("rows {rows:}, cols {cols:}");
//     let card = rows.min(cols) - (rows <= cols) as usize;
//     let mut proj: HouseholderReflection;
//     let mut w = vec![0f32; rows.max(cols)];
//     for o in 0..card {
//         println!("a {a:?}");
//         proj = householder_params(
//             // column vector
//             (o..rows).into_iter().map(|r| a.data[r * cols + o]).collect()
//         );
//         println!("here?");
//         // println!("w {w:?}");
        
//         // (I - bvv')A => w := bv'A
//         //  A -= vw'
//         for j in o..cols {
//             for i in o..rows {
//                 w[j] += proj.vector[i - o] * a.data[ i * cols + j];
//             }
//             w[j] *= proj.beta;
//             for i in o..rows {
//                 a.data[ i * cols + j] -= proj.vector[i - o] * w[j];
//             }
//             w[j] = 0_f32;
//         }
//         println!("there?");
//         // stop one early for columns because a[m,n-1] can be non-zero
//         if o+1 == card  { break; }
//         // [2..n]
//         // [3..n]
//         //
//         let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
//         println!("row_vector {row_vector:?}");
//         proj = householder_params(
//             // row vector
//             a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec()
//         );
//         println!("proj {proj:?}");

//         // A(I - bvv') => w = b * Av
//         // A -= wv'
//         for i in o+1..rows {
//             for j in o+1..cols {
//                 w[i] += a.data[i * cols + j] * proj.vector[j - o - 1];
//             }
//             w[i] *= proj.beta;
//         }
//         println!("w {w:?}");
//         for i in o+1..rows {
//             for j in o+1..cols {
//                 a.data[i * cols + j] -= w[i] * proj.vector[j - o - 1];
//             }
//             w[i] = 0_f32;
//         }
//     }
//     // TODO: uncomment so flows naturally into givens
//     let dims = rows.min(cols);
//     resize_rows(dims,&mut a); 
//     resize_cols(dims, &mut a); 
//     println!("a {a:?}");
//     a
// }

// // TODO: Optimize this method using the standard multiplication
// pub fn golub_kahan_explicit(mut a: NdArray) -> NdArray {
//     // TODO: refactor so householder_params mutates in place
//     let (rows, cols)  = (a.dims[0], a.dims[1]);
//     let card = rows.min(cols) - (rows <= cols) as usize;
//     let mut householder: HouseholderReflection;

//     let mut new: NdArray;
//     // 5, 2 // should be two
//     // 2, 5 should be one
//     // 3, 3 should be 2
//     for o in 0..card {
//         println!("a {a:?}");
//         new = create_identity_matrix(rows);
//         let column_vector = (o..rows)
//             .into_par_iter()
//             .map(|r| a.data[r * cols + o])
//             .collect::<Vec<f32>>();
//         householder = householder_params(column_vector);
//         if householder.beta < EPSILON {
//             continue;
//         }
//         println!("hello");
//         for i in 0..rows - o {
//             for j in 0..cols - o {
//                 new.data[(o + i) * cols + j + o] -=
//                     householder.beta * householder.vector[i] * householder.vector[j]
//             }

//         }
//         a = tensor_mult(4, &new, &a);
//         println!("hello");
//         // 4 
//         if o < card - 1 {
//             new = create_identity_matrix(cols);
//             // let row_vector: Vec<f32> = a.data[(o * cols) + 1..(o + 1) * cols].to_vec();
//             let row_vector: Vec<f32> = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
//             householder = householder_params(row_vector);

//             for i in 0..rows - o - 1 {
//                 for j in 0..cols - o - 1 {
//                     new.data[(o + i + 1) * cols + (j + o + 1)] -=
//                         householder.beta * householder.vector[i] * householder.vector[j];
//                 }
//             }
//             a = tensor_mult(4, &a, &new);
//         }
//     }
//     println!("a {a:?}");
//     a
// }
