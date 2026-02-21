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
use std::hint::black_box;
const CONVERGENCE_CONDITION: f32 = 1e-4;

// TODO: 
// optimize householder instantiation
// derive outer product derivation for cholesky
// rank 1 lu updates for lp
// golub kahan, remove the w[i]
// perhaps redefine householder to borrow
// compilation for simd optimization in like simd matrix mult


fn main() {
    let n = 1000;
    let mut x = generate_random_matrix(n, n);
    // println!("x {x:?}");
    let start = Instant::now();
    for _ in 0..100 {
        let ksvd = RandomizedSvd::new(20, x.clone());
        let tiny = ksvd.approx();
        let big = ksvd.reconstruct();
        black_box(tiny);
        black_box(big);
        black_box(&x);
        // let svalues = RankKSvd::new(4, x.clone());
    }
    let duration = start.elapsed();
    println!("Pipeline took {:?}", duration / 100);

}
