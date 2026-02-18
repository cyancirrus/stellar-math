#![allow(dead_code, unused_imports)]
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::qr::qr_decompose;
use stellar::decomposition::svd::golub_kahan;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};
use stellar::structure::ndarray::NdArray;
use stellar::solver::randomized_svd::{RankKSvd, RandomizedSvd};

const CONVERGENCE_CONDITION: f32 = 1e-4;

fn main() {
    let n = 6;
    let mut x = generate_random_matrix(n, n);
    println!("x {x:?}");
    let ksvd = RandomizedSvd::new(4, x.clone());

    ksvd.qrl.left_apply_qt(&mut x);
    x = x.transpose();
    ksvd.qrr.left_apply_qt(&mut x);
    x = x.transpose();
    println!("rotated {x:?}");

    let tiny = ksvd.approx();
    println!("tiny {tiny:?}");

    let big = ksvd.reconstruct();
    println!("big {big:?}");

}
