#![allow(dead_code, unused_imports, unused_mut, unused_variables)]
use std::time::Instant;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::qr::QrDecomposition;
use stellar::decomposition::svd::golub_kahan;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};
use stellar::solver::randomized_svd::{RandomizedSvd, RankKSvd};
use stellar::structure::ndarray::NdArray;

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

fn main() {
    let n = 6;
    let mut x = generate_random_matrix(n, n);
    println!("x {x:?}");
    let start = Instant::now();
    let ksvd = RandomizedSvd::new(6, x.clone());

    ksvd.qrl.left_apply_qt(&mut x);
    x = x.transpose();
    ksvd.qrr.left_apply_qt(&mut x);
    x = x.transpose();
    let tiny = ksvd.approx();
    let big = ksvd.reconstruct();
    let svalues = RankKSvd::new(4, x.clone());
    let duration = start.elapsed();
    // println!("Pipeline took {:?}", duration / 100);

    println!("rotated {x:?}");
    println!("tiny {tiny:?}");
    println!("big {big:?}");
    println!("s reference {:?}", ksvd.svd.s);
    println!("singular values {:?}", svalues.singular);
}
