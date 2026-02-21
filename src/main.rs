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
use stellar::algebra::ndmethods::{create_identity_matrix, matrix_mult, transpose};
use std::hint::black_box;
const CONVERGENCE_CONDITION: f32 = 1e-4;

// TODO: 
// optimize householder instantiation
// derive outer product derivation for cholesky
// rank 1 lu updates for lp
// golub kahan, remove the w[i]
// perhaps redefine householder to borrow
// compilation for simd optimization in like simd matrix mult

fn checking_qr(k:usize, mut matrix: NdArray) {
        let n = matrix.dims[0];
        let sketch = generate_random_matrix(n, k);
        // might wish to inner product the resulting matrix
        // n x k
        let a_sketch = matrix_mult(&matrix, &sketch);
        // implicit covariance
        let y = matrix_mult(&matrix, &matrix_mult(&matrix.transpose(), &a_sketch));
        // left ortho
        let qrl = QrDecomposition::new(y);
        qrl.left_apply_qt(&mut matrix);
        let mut tiny_core = matrix.transpose();
        let qrr = QrDecomposition::new(tiny_core.clone());
        qrr.left_apply_qt(&mut tiny_core);
        tiny_core.transpose_square();
}


fn main() {
    let n = 1000;
    let mut x = generate_random_matrix(n, n);
    // println!("x {x:?}");
    let start = Instant::now();
    for _ in 0..100 {
        let ksvd = checking_qr(20, x.clone());
        // let tiny = ksvd.approx();
        // let big = ksvd.reconstruct();
        black_box(ksvd);
    }
    let duration = start.elapsed();
    println!("Pipeline took {:?}", duration / 100);

}
