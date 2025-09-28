use crate::algebra::ndmethods::create_identity_matrix;
use crate::algebra::ndmethods::tensor_mult;
use crate::decomposition::qr::qr_decompose;
use crate::decomposition::lu::lu_decompose;
use crate::structure::ndarray::NdArray;

use crate::random::generation::{generate_random_matrix, generate_random_vector, generate_random_symetric};

const CONVERGENCE_CONDITION: f32 = 1e-4;

fn rayleigh_inverse_iteration(mut matrix: NdArray) -> Vec<f32> {
    // (A - Iu)y = x;
    // x' = y/||y||;
    // let M := (A-Iu)
    // LU(M) -> solve => y
    // u' := rayleigh quotient of x
    debug_assert!(matrix.dims.len() == 2);
    debug_assert!(matrix.dims[0] == matrix.dims[1]);
    let n = matrix.dims[0];
    let mut current = generate_random_vector(n);
    let mut u = 0_f32;
    let mut error = 1_f32;
    while CONVERGENCE_CONDITION < error {
        // transforms M' = (A - Iu + Iu - Iu')
        let previous = current.clone();
        estimate_eigenvalues(&mut u, &mut matrix, &current);
        let lu = lu_decompose(matrix.clone());
        // eigen is now y
        lu.solve_inplace_vec(&mut current);
        normalize_vector(&mut current);
        error = vector_diff_norm(&current, &previous);
        println!("error: {error:?}");
    }
    current
}

fn vector_diff_norm(a: &[f32], b: &[f32]) -> f32 {
    // distance :: SS (sign*a[ij] - b[ij])^2
    // sign := a'b
    debug_assert!(a.len() == b.len());
    let n = a.len();
    let mut error = 0_f32;
    for k in 0..n {
        let diff = a[k] - b[k];
        error += diff * diff;
    }
    (error / (n as f32)).sqrt()
    // error.sqrt() / (n as f32)
}

fn normalize_vector(x: &mut [f32]) {
    let n = x.len();
    let mut norm = 0_f32;
    for i in 0..n {
        norm += x[i] * x[i];
    }
    norm = norm.sqrt();
    for val in x {
        *val /= norm;
    }
}
    

fn frobenius_diff_norm(a: &NdArray, b: &NdArray) -> f32 {
    // distance :: SS (sign*a[ij] - b[ij])^2
    // sign := a'b
    debug_assert!(a.dims == b.dims);
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let mut error = 0_f32;
    for j in 0..cols {
        for i in 0..rows {
            let diff = a.data[i * cols + j] - b.data[i * cols + j];
            error += diff * diff;
        }
    }
    error.sqrt()
}

fn estimate_eigenvalues(u:&mut f32, a: &mut NdArray, x:&[f32]) {
    // estimated via rayleigh quotient
    // x'Ax/x'x
    debug_assert_eq!(a.dims[0], a.dims[1]); 
    let n = a.dims[0];
    // center M to A ->  A-u +u = A
    for i in 0..n {
        a.data[i * n + i] += *u;
    }
    // only desire the diagonal
    let mut w = vec![0f32;n];
    let mut numerator = 0_f32;
    let mut denominator = 0_f32;
    for i in 0..n {
        for k in 0..n {
            w[i] +=  a.data[ i * n + k] * x[k];
        }
    }
    for k in 0..n {
        numerator += w[k] * x[k];
        denominator += x[k] * x[k];
    }
    *u = numerator / denominator;
    // perterb M by new uii : A - u' 
    for d in 0..n {
        a.data[d * n + d] -= *u;
    }
}


// fn test_random_eigenvector() {
//     let n = 128;
//     // Step 1: create a random symmetric matrix
//     let matrix = generate_random_symetric(n);

//     // Step 2: run your single-vector eigenvector iteration
//     let eigenvec = rayleigh_inverse_iteration(matrix.clone()); // now returns NdArray with shape [n]

//     // Step 3: check that it's normalized (||v|| = 1)
//     let norm: f32 = eigenvec.iter().map(|x| x * x).sum::<f32>().sqrt();
//     assert!((norm - 1.0).abs() < 1e-3, "Vector not normalized");

//     // Step 4: check eigenvector property A v ~ lambda v
//     let mut a_v = vec![0.0; n];
//     for i in 0..n {
//         for j in 0..n {
//             a_v[i] += matrix.data[i * n + j] * eigenvec[j];
//         }
//     }

//     // estimate lambda using Rayleigh quotient
//     let lambda: f32 = a_v
//         .iter()
//         .zip(eigenvec.iter())
//         .map(|(av, v)| av * v)
//         .sum::<f32>()
//         / eigenvec.iter().map(|v| v * v).sum::<f32>();

//     // check A v ~ lambda v
//     for i in 0..n {
//         let diff = (a_v[i] - lambda * eigenvec[i]).abs();
//         assert!(
//             diff < 1e-3,
//             "Eigenvector property failed at index {}: diff={}",
//             i,
//             diff
//         );
//     }

//     println!("Test passed! Eigenvalue estimate: {}", lambda);
// }


// fn main() {
//     // it's in proto.bu
//     test_random_eigenvector();
// }

