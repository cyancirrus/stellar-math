use crate::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
use crate::decomposition::givens::{SingularValueDecomp, full_givens_iteration, givens_iteration};
use crate::decomposition::qr_matrix::QrDecomp;
use crate::decomposition::svd::{full_golub_kahan, golub_kahan};
use crate::random::generation::generate_random_matrix;
use crate::structure::ndarray::NdArray;

pub struct RandomizedSvd {
    pub n: usize,
    pub k: usize,
    pub qrl: QrDecomp,
    pub qrr: QrDecomp,
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
        // n x k
        // implicit covariance
        let y = matrix_mult(
            &matrix,
            &lt_matrix_mult(&matrix, &matrix_mult(&matrix, &sketch)),
        );

        let qrl = QrDecomp::new(y);
        qrl.left_apply_qt(&mut matrix);
        let reference = golub_kahan(matrix);
        let singular = givens_iteration(reference);
        Self { n, k, singular }
    }
}

impl RandomizedSvd {
    pub fn new(k: usize, mut matrix: NdArray) -> Self {
        let n = matrix.dims[0];
        let sketch = generate_random_matrix(n, k);
        // implicit covariance
        let y = matrix_mult(
            &matrix,
            &lt_matrix_mult(&matrix, &matrix_mult(&matrix, &sketch)),
        );
        let qrl = QrDecomp::new(y);
        qrl.left_apply_qt(&mut matrix);
        matrix.resize_rows(k);
        matrix.transpose_inplace();
        let qrr = QrDecomp::new(matrix);
        let mut tiny_core = qrr.t.clone();
        tiny_core.resize_rows(k);
        tiny_core.transpose_square();
        let (u, b, v) = full_golub_kahan(tiny_core);
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
            for k in 0..self.k {
                for j in 0..self.k {
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
        let mut tiny = vec![0_f32; self.k * self.n];
        for i in 0..self.k {
            for k in 0..self.k {
                let lambda = self.svd.s.data[k * self.k + k];
                let t_row = &mut tiny[i * self.n..(i + 1) * self.n];
                let u_i = self.svd.u.data[i * self.k + k];
                for j in 0..self.k {
                    t_row[j] += u_i * lambda * self.svd.v.data[j * self.k + k];
                }
            }
        }
        NdArray {
            dims: vec![self.k, self.n],
            data: tiny,
        }
    }
    pub fn reconstruct(&self) -> NdArray {
        let mut output = self.approx_padded();
        self.qrr.right_apply_qt(&mut output);
        output.resize_rows(self.n);
        self.qrl.left_apply_q(&mut output);
        output
    }
}

// use std::hint::black_box;
// use std::time::Instant;
// use stellar::algebra::ndmethods::tensor_mult;
// use stellar::decomposition::givens::{givens_iteration, SingularValueDecomp};
// use stellar::decomposition::lower_upper::LuPivotDecompose;
// use stellar::decomposition::qr::QrDecomposition;
// use stellar::decomposition::svd::golub_kahan;
// use stellar::random::generation::{generate_random_matrix, generate_random_symetric};
// use stellar::structure::ndarray::NdArray;
// use stellar::solver::randomized_svd::{RankKSvd, RandomizedSvd};

// fn main() {
//     let n = 1000;
//     let mut x = generate_random_matrix(n, n);
//     // println!("x {x:?}");
//     let start = Instant::now();
//     for _ in 0..100 {
//         let ksvd = RandomizedSvd::new(20, x.clone());
//         let tiny = ksvd.approx();
//         let big = ksvd.reconstruct();
//         black_box(tiny);
//         black_box(big);
//         black_box(&x);
//         // let svalues = RankKSvd::new(4, x.clone());
//     }
//     let duration = start.elapsed();
//     println!("Pipeline took {:?}", duration / 100);

// }
