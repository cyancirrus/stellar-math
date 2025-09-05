use stellar::decomposition::svd::golub_kahan_explicit;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::qr::qr_decompose;
use stellar::decomposition::givens::givens_iteration;
use stellar::structure::ndarray::NdArray;
use stellar::decomposition::householder::householder_factor;


struct Cholesky {
    l: NdArray,
}


fn cholesky(a:NdArray) -> Cholesky {
    // update lij
    // aii = lik * lik + lii^2
    // => lii = (aii - sum 0..i lik)^0.5
    // aij = Sum 0..j lik * ljk + lij * ljj
    // => ljj = 1/ ljj * (aij - Sum 0..j lik * ljk )
    // a32 = l31 * l21 + l32 * l22
    let rows = a.dims[0];
    let cols = a.dims[1];
    debug_assert_eq!(rows, cols);
    let mut l = vec![0_f32; rows * cols];

    for i in 0..rows {
        for j in 0..=i {
            let mut sum = 0_f32;
            for k in 0..j {
                sum += l[i*cols + k] * l[j*cols + k];
            }
            if i==j { l[i * cols + i] = (a.data[i*cols + i] - sum).sqrt(); }
            else { l[i * cols + j] = (a.data[i * cols + j] - sum) / l[j * cols + j]};

        }
    }
    Cholesky {
        l: NdArray {
            dims: a.dims.clone(),
            data:l,
        }
    }

}



fn main() {
}

