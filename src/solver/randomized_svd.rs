use crate::decomposition::qr::{qr_decompose};
use crate::decomposition::lu::{lu_decompose};
use crate::decomposition::svd::{golub_kahan};
use crate::decomposition::givens::{givens_iteration, SingularValueDecomp};
use crate::structure::ndarray::NdArray;
use crate::algebra::ndmethods::tensor_mult;
use crate::random::generation::{generate_random_matrix, generate_random_symetric};

const CONVERGENCE_CONDITION: f32 = 1e-4;


pub fn randomized_svd(k:usize, mut matrix:NdArray) -> SingularValueDecomp {
    let n = matrix.dims[0];
    let sketch = generate_random_matrix(n, k);
    // might wish to inner product the resulting matrix
    let cov = tensor_mult(4, &matrix , &matrix.transpose());
    let a_sketch = tensor_mult(4, &matrix, &sketch);
    let y = tensor_mult(4, &cov, &a_sketch);

    let qr = qr_decompose(y);
    qr.left_apply_qt(&mut matrix);
    let reference = golub_kahan(matrix);
    givens_iteration(reference)
}


// fn main() {
//     let n = 10;
//     let matrix = generate_random_matrix(n, n);
//     let x = matrix.clone();
//     let bidiag_reference = golub_kahan(x);
//     println!("bidiag reference {bidiag_reference:?}");
//     let svd_reference = givens_iteration(bidiag_reference);
//     println!("svd_reference u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}",svd_reference.u, svd_reference.s, svd_reference.v);
     
//     let x = matrix.clone();
//     let svd_randomized = randomized_svd(4, x);
//     println!("svd_randomized u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}",svd_randomized.u, svd_randomized.s, svd_randomized.v);
// }
