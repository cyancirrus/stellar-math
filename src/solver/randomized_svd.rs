use crate::algebra::ndmethods::matrix_mult;
use crate::decomposition::givens::{full_givens_iteration, givens_iteration, SingularValueDecomp};
use crate::decomposition::lower_upper::LuPivotDecompose;
use crate::decomposition::qr::QrDecomposition;
use crate::decomposition::svd::{full_golub_kahan, golub_kahan};
use crate::random::generation::{generate_random_matrix, generate_random_symetric};
use crate::structure::ndarray::NdArray;

const CONVERGENCE_CONDITION: f32 = 1e-4;

pub struct RandomizedSvd {
    pub n: usize,
    pub k: usize,
    pub qrl: QrDecomposition,
    pub qrr: QrDecomposition,
    pub svd: SingularValueDecomp,
}

pub struct RankKSvd {
    pub n: usize,
    pub k: usize,
    pub singular: Vec<f32>,
}

impl RankKSvd {
    // singular vlaues are up to sign convention
    // ie if l[k] < 0 => sign flip u[k]
    pub fn new(k: usize, mut matrix: NdArray) -> Self {
        let n = matrix.dims[0];
        let sketch = generate_random_matrix(n, k);
        // might wish to inner product the resulting matrix
        // let cov = matrix_mult(&matrix, &matrix.transpose());
        // n x k
        let a_sketch = matrix_mult(&matrix, &sketch);
        // implicit covariance
        let y = matrix_mult(&matrix, &matrix_mult(&matrix.transpose(), &a_sketch));

        let qrl = QrDecomposition::new(y);
        qrl.left_apply_qt(&mut matrix);
        let reference = golub_kahan(matrix);
        let singular = givens_iteration(reference);
        Self { n, k, singular }
    }
}

impl RandomizedSvd {
    // apply the transformations to u, v prior to storing requires
    // refactor of golub kahan
    pub fn new(k: usize, mut matrix: NdArray) -> Self {
        let n = matrix.dims[0];
        let sketch = generate_random_matrix(n, k);
        // might wish to inner product the resulting matrix
        // let cov = matrix_mult(&matrix, &matrix.transpose());
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
        let (u, b, v) = full_golub_kahan(tiny_core.transpose());
        let svd = full_givens_iteration(u, b, v);
        RandomizedSvd {
            n,
            k,
            qrl,
            qrr,
            svd,
        }
    }
    pub fn approx(&self) -> NdArray {
        let mut tiny = vec![0_f32; self.k * self.k];
        for i in 0..self.k {
            for j in 0..self.k {
                for k in 0..self.k {
                    tiny[i * self.k + j] += self.svd.u.data[i * self.k + k]
                        * self.svd.s.data[k * self.k + k]
                        * self.svd.v.data[j * self.k + k];
                }
            }
        }
        NdArray {
            dims: vec![self.k, self.k],
            data: tiny,
        }
    }
    fn approx_padded(&self) -> NdArray {
        let mut tiny = vec![0_f32; self.n * self.n];
        for i in 0..self.k {
            for j in 0..self.k {
                for k in 0..self.k {
                    tiny[i * self.n + j] += self.svd.u.data[i * self.k + k]
                        * self.svd.s.data[k * self.k + k]
                        * self.svd.v.data[j * self.k + k];
                }
            }
        }
        NdArray {
            dims: vec![self.n, self.n],
            data: tiny,
        }
    }
    pub fn reconstruct(&self) -> NdArray {
        let mut output = self.approx_padded();
        self.qrr.right_apply_qt(&mut output);
        self.qrl.left_apply_q(&mut output);
        output
    }
}

// #![allow(dead_code, unused_imports)]
// use stellar::algebra::ndmethods::tensor_mult;
// use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
// use stellar::decomposition::lower_upper::LuPivotDecompose;
// use stellar::decomposition::qr::QrDecomposition;
// use stellar::decomposition::svd::golub_kahan;
// use stellar::random::generation::{generate_random_matrix, generate_random_symetric};
// use stellar::structure::ndarray::NdArray;
// use stellar::solver::randomized_svd::{RankKSvd, RandomizedSvd};

// const CONVERGENCE_CONDITION: f32 = 1e-4;

// fn main() {
//     let n = 1000;
//     let mut x = generate_random_matrix(n, n);
//     println!("x {x:?}");
//     let start = Instant::now();
//     for _ in 0..100 {
//         let ksvd = RandomizedSvd::new(20, x.clone());

//         ksvd.qrl.left_apply_qt(&mut x);
//         x = x.transpose();
//         ksvd.qrr.left_apply_qt(&mut x);
//         x = x.transpose();
//         let tiny = ksvd.approx();
//         let big = ksvd.reconstruct();
//         let svalues = RankKSvd::new(4, x.clone());
//     }
//     let duration = start.elapsed();
//     println!("Pipeline took {:?}", duration / 100);

// //     println!("rotated {x:?}");
// //     println!("tiny {tiny:?}");
// //     // println!("big {big:?}");
// //     println!("s reference {:?}", ksvd.svd.s);
// //     println!("singular values {:?}", svalues.singular);

// }
