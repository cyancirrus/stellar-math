#![allow(dead_code, unused_imports, unused_mut, unused_variables)]
use std::time::Instant;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::qr::QrDecomposition;
use stellar::decomposition::svd::golub_kahan;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};
use stellar::solver::randomized_svd::{RandomizedSvd, RankKSvd};
use stellar::structure::ndarray::NdArray;
use stellar::equality::approximate::approx_vector_eq;

const CONVERGENCE_CONDITION: f32 = 1e-4;

// TODO: optimize qr applies
// create apply_q_right for qr
// create apply_qt_right for qr
// optimize householder instantiation
// derive outer product derivation for cholesky
// rank 1 lu updates for lp
// golub kahan, remove the w[i]
// perhaps redefine householder to borrow
// compilation for simd optimization in like simd matrix mult

fn test_qr_right_apply_t() {
    let n = 6;
    let x = generate_random_matrix(n, n);
    let qr = QrDecomposition::new(x);
    let mut y = generate_random_matrix(n, n);
    let y_clone = y.clone();
    qr.right_apply_q(&mut y);
    qr.right_apply_qt(&mut y);
    assert!(approx_vector_eq(&y.data, &y_clone.data));
    //-----------------------------------
    let n = 6;
    let x = generate_random_matrix(n, n);
    let qr = QrDecomposition::new(x);
    let mut y = generate_random_matrix(n, n);
    let y_clone = y.clone();
    qr.right_apply_qt(&mut y);
    qr.right_apply_q(&mut y);
    assert!(approx_vector_eq(&y.data, &y_clone.data));
}

fn test_qr_left_apply_t() {
    let n = 6;
    let x = generate_random_matrix(n, n);
    let qr = QrDecomposition::new(x);
    let mut y = generate_random_matrix(n, n);
    let y_clone = y.clone();
    qr.left_apply_q(&mut y);
    qr.left_apply_qt(&mut y);
    assert!(approx_vector_eq(&y.data, &y_clone.data));
    //-----------------------------------
    let n = 6;
    let x = generate_random_matrix(n, n);
    let qr = QrDecomposition::new(x);
    let mut y = generate_random_matrix(n, n);
    let y_clone = y.clone();
    qr.left_apply_qt(&mut y);
    qr.left_apply_q(&mut y);
    assert!(approx_vector_eq(&y.data, &y_clone.data));
}

fn main() {
    test_qr_right_apply_t();
    test_qr_left_apply_t();
}
