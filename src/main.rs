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
    let n = 10;
    let mut x = generate_random_matrix(n, n);
    println!("x {x:?}");
    let ksvd = RandomizedSvd::new(4, x.clone());
    // println!("approx {:?}", ksvd.reconstruct());

    ksvd.qrl.left_apply_qt(&mut x);
    x = x.transpose();
    ksvd.qrr.left_apply_qt(&mut x);
    x = x.transpose();
    println!("final x {x:?}");

    let tiny = ksvd.approx();
    println!("tiny x {tiny:?}");

        // println!("svd_randomized u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}", ksvd.svd.u, ksvd.svd.s, ksvd.svd.v);

}

// fn main() {
//     let n = 10;
//     let matrix = generate_random_matrix(n, n);
//     let x = matrix.clone();
//     println!("x {x:?}");
//     let bidiag_reference = golub_kahan(x);
//     println!("bidiag reference {bidiag_reference:?}");
//     let svd_reference = givens_iteration(bidiag_reference);
//     println!("svd_reference u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}",svd_reference.u, svd_reference.s, svd_reference.v);

//     let x = matrix.clone();
//     let ksvd = RankKSvd::new(6, x);
//     println!("svd_randomized u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}", ksvd.svd.u, ksvd.svd.s, ksvd.svd.v);
//     // println!("approx {:?}", ksvd.reconstruct());
// }
