use crate::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
use crate::decomposition::givens::{SingularValueDecomp, full_givens_iteration, givens_iteration};
use crate::decomposition::qr_matrix::QrDecomp;
use crate::decomposition::svd::{full_golub_kahan, golub_kahan};
use crate::random::generation::generate_random_matrix;
use crate::structure::ndarray::NdArray;

// NOTE: should be able to left apply only up to k

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
        let mut tiny = vec![0f32; self.k * self.k];
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
        let mut tiny = vec![0f32; self.k * self.n];
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
        // 7ms
        let mut output = self.approx_padded();
        self.qrr.right_apply_qt(&mut output);
        output.resize_rows(self.n);
        self.qrl.left_apply_q(&mut output);
        output
    }
}

// use std::hint::black_box;
// use std::time::{Duration, Instant};
// use stellar::random::generation::generate_random_matrix;
// use stellar::solver::randomized_svd::RandomizedSvd;

// fn main() {
//     let n = 1000;
//     let k = 20;
//     let iterations = 100;
//     let mut total = Duration::ZERO;

//     for _ in 0..iterations {
//         let x = generate_random_matrix(n, n);
//         let x_for_svd = x.clone();

//         let start = Instant::now();

//         let ksvd = RandomizedSvd::new(k, x_for_svd);
//         let tiny = ksvd.approx();
//         let big = ksvd.reconstruct();

//         total += start.elapsed();
//         black_box(tiny);
//         black_box(big);
//         black_box(x);
//     }

//     println!("Average Pipeline took: {:?}", total / iterations as u32);
// }
