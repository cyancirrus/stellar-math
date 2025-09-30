// #[cfg(target_arch = "x86_64")]

// reading : https://en.wikipedia.org/wiki/Schur_complement
// when writing article write something about recursive / iterative defns and which info is
// available, ie why need to reverse iteration on QR


// move code into examples directory
// cargo run --example demo
use stellar::decomposition::qr::{qr_decompose};
use stellar::decomposition::lu::{lu_decompose};
use stellar::decomposition::svd::{golub_kahan};
use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
use stellar::structure::ndarray::NdArray;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};

const CONVERGENCE_CONDITION: f32 = 1e-4;


fn randomized_svd(k:usize, mut matrix:NdArray) -> SingularValueDecomp {
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
//     let n = 4;
//     let matrix = generate_random_matrix(n, n);
//     golub_kahan(matrix);
// }

fn main() {
    let n = 10;
    let matrix = generate_random_matrix(n, n);
    // let mut data = vec![0_f32; 9];
    // let dims = vec![3; 2];
    // data[0] = 1_f32;
    // data[1] = 2_f32;
    // data[2] = 3_f32;
    // data[3] = 5_f32;
    // data[4] = 4_f32;
    // data[5] = 5_f32;
    // data[6] = 6_f32;
    // data[7] = 2_f32;
    // data[8] = 8_f32;

    // let matrix = NdArray::new(dims, data);
    let x = matrix.clone();
    let bidiag_reference = golub_kahan(x);
    println!("bidiag reference {bidiag_reference:?}");
    let svd_reference = givens_iteration(bidiag_reference);
    println!("svd_reference u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}",svd_reference.u, svd_reference.s, svd_reference.v);
     
    let x = matrix.clone();
    let svd_randomized = randomized_svd(4, x);
    println!("svd_randomized u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}",svd_randomized.u, svd_randomized.s, svd_randomized.v);
    
}
