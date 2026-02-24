#![allow(unused_variables, dead_code, unused_imports, unused_mut)]
use stellar::decomposition::qr_matrix::QrDecomp;
use std::hint::black_box;
use std::time::Instant;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::algebra::ndmethods::{create_identity_matrix, matrix_mult, transpose};
use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::qr::QrDecomposition;
use stellar::decomposition::svd::golub_kahan;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};
use stellar::solver::randomized_svd::{RandomizedSvd, RankKSvd};
use stellar::structure::ndarray::NdArray;
const CONVERGENCE_CONDITION: f32 = 1e-4;


fn check_householder_matrix() {
    let n = 5;
    let x = generate_random_matrix(n, n);
    println!("X {x:?}");
    let qr_old = QrDecomposition::new(x.clone());
    let qr_new = QrDecomp::new(x.clone());
    let y = generate_random_matrix(n, n);
    // let y = x.transpose();
    let mut y_expect= y.clone();
    let mut y_actual = y.clone();
    qr_old.right_apply_q(&mut y_expect);
    qr_new.right_apply_q(&mut y_actual);

    println!("Y_expected {y_expect:?}");
    println!("Y_actual {y_actual:?}");

}
fn main() {
    check_householder_matrix();
}
