#![allow(dead_code, unused_imports)]
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::qr::qr_decompose;
use stellar::decomposition::svd::golub_kahan;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};
use stellar::structure::ndarray::NdArray;
use stellar::solver::randomized_svd::{RankKSvd, RandomizedSvd};
use std::time::Instant;

const CONVERGENCE_CONDITION: f32 = 1e-4;


// fn embed_givens(n: usize, i: usize, j: usize, c: f32, s: f32) -> NdArray {
//     let mut array = create_identity_matrix(n);
//     array.data[i * n + i] = c;
//     array.data[i * n + j] = s;
//     array.data[j * n + i] = -s;
//     array.data[j * n + j] = c;
//     array
// }

// fn test_givens() {
//     let a = generate_random_matrix(5, 5);
    


// }


fn main() {
    let n = 8;
    let mut x = generate_random_matrix(n, n);
    println!("x {x:?}");
    let start = Instant::now();
    let ksvd = RandomizedSvd::new(6, x.clone());

    ksvd.qrl.left_apply_qt(&mut x);
    x = x.transpose();
    ksvd.qrr.left_apply_qt(&mut x);
    x = x.transpose();
    let tiny = ksvd.approx();
    let _big = ksvd.reconstruct();
    let svalues = RankKSvd::new(4, x.clone());
    let duration = start.elapsed();
    println!("Pipeline took {:?}", duration / 100);
    println!("rotated {x:?}");
    println!("tiny {tiny:?}");
    // println!("big {big:?}");
    println!("s reference {:?}", ksvd.svd.s);
    println!("singular values {:?}", svalues.singular);



}
