use crate::algebra::ndmethods::matrix_mult;
use crate::decomposition::givens::{givens_iteration, full_givens_iteration, SingularValueDecomp};
use crate::decomposition::lower_upper::LuPivotDecompose;
use crate::decomposition::qr::{qr_decompose, QrDecomposition};
use crate::decomposition::svd::{golub_kahan, full_golub_kahan};
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
    pub svd: SingularValueDecomp,
}

impl RankKSvd {
    // singular vlaues are up to sign convention
    // ie if l[k] < 0 => sign flip u[k]
    pub fn new(k: usize, mut matrix: NdArray) -> Self {
        let n = matrix.dims[0];
        let sketch = generate_random_matrix(n, k);
        // might wish to inner product the resulting matrix
        let cov = matrix_mult(&matrix, &matrix.transpose());
        // n x k
        let a_sketch = matrix_mult(&matrix, &sketch);
        let y = matrix_mult(&cov, &a_sketch);

        let qrl = qr_decompose(y);
        qrl.left_apply_qt(&mut matrix);
        let reference = golub_kahan(matrix);
        let svd =  givens_iteration(reference);
        Self {
            n,
            k,
            svd
        }
    }
    pub fn approx(&self) -> NdArray {
        let mut tiny= vec![0_f32 ; self.k * self.k ];
        for i in 0..self.k {
            for j in 0..self.k {
                for k in 0..self.k {
                    tiny[i * self.k  + j] += self.svd.u.data[ i * self.k + k]
                        * self.svd.s.data[ k * self.k + k]
                        * self.svd.v.data[ k * self.k + j];
                }
            }
        }
        NdArray {
            dims: vec![self.k, self.k],
            data: tiny,
        }
    }
}

impl RandomizedSvd {
    // apply the transformations to u, v prior to storing requires
    // refactor of golub kahan
    pub fn new(k: usize, mut matrix: NdArray) -> Self {
        let n = matrix.dims[0];
        let sketch = generate_random_matrix(n, k);
        // might wish to inner product the resulting matrix
        let cov = matrix_mult(&matrix, &matrix.transpose());
        // n x k
        let omega = matrix_mult(&matrix, &sketch);
        let y = matrix_mult(&cov, &omega);
        // left ortho
        let qrl = qr_decompose(y);
        qrl.left_apply_qt(&mut matrix);
        let mut tiny_core = matrix.transpose();
        let qrr = qr_decompose(tiny_core.clone());
        qrr.left_apply_qt(&mut tiny_core);
        println!("fitting matrix {:?}", tiny_core.transpose());
        let (u, b, v) = full_golub_kahan(tiny_core.transpose());
        let svd =  full_givens_iteration(u, b, v);
        RandomizedSvd {
            n,
            k,
            qrl,
            qrr,
            svd
        }
    }
    pub fn approx(&self) -> NdArray {
        let mut tiny= vec![0_f32 ; self.k * self.k ];
        for i in 0..self.k {
            for j in 0..self.k {
                for k in 0..self.k {
                    tiny[i * self.k  + j] += self.svd.u.data[ i * self.k + k]
                        * self.svd.s.data[ k * self.k + k]
                        * self.svd.v.data[ j * self.k + k];
                }
            }
        }
        NdArray {
            dims: vec![self.k, self.k],
            data: tiny,
        }
    }
    pub fn reconstruct(&self) -> NdArray {
        let mut output = self.approx();
        output.data.resize(self.n * self.k, 0_f32);
        output.dims[0] = self.n;
        self.qrl.left_apply_q(&mut output);
        output = output.transpose();
        output.data.resize(self.n * self.n, 0_f32);
        output.dims[0] = self.n;
        self.qrr.left_apply_q(&mut output);
        output = output.transpose();
        output
    }
}
