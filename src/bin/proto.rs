use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::qr::{qr_decompose, QrDecomposition};
use stellar::decomposition::lu::{lu_decompose, LuDecomposition};
use stellar::structure::ndarray::NdArray;

// #[cfg(target_arch = "x86_64")]
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_distr::Normal;
use rand_distr::StandardNormal;

// reading : https://en.wikipedia.org/wiki/Schur_complement
// when writing article write something about recursive / iterative defns and which info is
// available, ie why need to reverse iteration on QR


// move code into examples directory
// cargo run --example demo
const CONVERGENCE_CONDITION: f32 = 1e-6;

fn rayleigh_inverse_iteration(mut matrix: NdArray) -> NdArray {
    // (A - Iu)y = x;
    // x' = Q from QR(y)
    // let M := (A-Iu)
    // LU(M) -> solve => y
    // u' := rayleigh quotient of x
    debug_assert!(matrix.dims.len() == 2);
    debug_assert!(matrix.dims[0] == matrix.dims[1]);
    let n = matrix.dims[0];
    let mut current = generate_random_matrix(n, n);
    let mut u = vec![0_f32;n];
    let mut error = 1_f32;
    while CONVERGENCE_CONDITION < error {
        // transforms M' = (A - Iu + Iu - Iu')
        let previous = current.clone();
        estimate_eigenvalues(&mut u, &mut matrix, &current);
        let lu = lu_decompose(matrix.clone());
        // eigen is now y
        lu.solve_inplace(&mut current);
        let qr = qr_decompose(current.clone());
        current = qr.projection_matrix();
        error = frobenius_diff_norm(&current, &previous);
        println!("error: {error:?}");
    }
    current
}
    
fn generate_random_matrix(m:usize, n:usize) -> NdArray {
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

fn estimate_eigenvalues(u:&mut[f32], a: &mut NdArray, x:&NdArray) {
    // estimated via rayleigh quotient
    // x'Ax/x'x
    debug_assert_eq!(a.dims[0], a.dims[1]); 
    let n = a.dims[0];
    // center M to A ->  A-u +u = A
    for i in 0..n {
        a.data[i * n + i] += u[i];
    }
    // only desire the diagonal
    for d in 0..n {
        let mut w = vec![0f32;n];
        let mut numerator = 0_f32;
        let mut denominator = 0_f32;
        for i in 0..n {
            for k in 0..n {
                w[i] +=  a.data[ i * n + k] * x.data[k * n + d];
            }
        }
        for k in 0..n {
            numerator += w[k] * x.data[k * n + d];
            denominator += x.data[k * n + d] * x.data[k * n + d];
        }
        u[d] = numerator / denominator;
        // perterb M by new uii : A - u' 
        a.data[d * n + d] -= u[d];
    }
}



// fn eigen_iteration(matrix: NdArray) -> NdArray {
//     debug_assert!(matrix.dims.len() == 2);
//     debug_assert!(matrix.dims[0] == matrix.dims[1]);
//     let mut eigen = generate_random_matrix(&matrix.dims);
//     let mut error = 1_f32;
//     while CONVERGENCE_CONDITION < error {
//         let next = tensor_mult(4, &matrix, &eigen);
//         let qr = qr_decompose(next);
//         let projection = qr.projection_matrix();
//         error = frobenius_norm(&projection, &eigen);
//         println!("error: {error:?}");
//         eigen = projection;
//     }
//     eigen
// }


fn main() {
    // it's in proto.bu
    // test_random_eigenvectors();
}
