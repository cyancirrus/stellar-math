use crate::decomposition::qr::QrDecomposition;
use crate::decomposition::schur::real_schur;
use crate::decomposition::svd::golub_kahan;
use crate::structure::ndarray::NdArray;

// Tihnov
// https://en.wikipedia.org/wiki/Ridge_regression

// Learn runge kutta soon
// https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

const EPSILON: f32 = 1e-6;

fn scale_identity(n: usize, c: f32) -> NdArray {
    let mut mat = vec![0_f32; n * n];
    for i in 0..n {
        mat[i * n + i] = c;
    }
    NdArray::new(vec![n, n], mat)
}

fn target_rotation_indices(matrix: &NdArray) -> (usize, usize) {
    debug_assert!(matrix.dims.len() == 2);
    debug_assert!(matrix.dims[0] == matrix.dims[1]);
    let m = matrix.dims[0];
    let mut rswap = m - 1;
    for i in 0..m {
        if matrix.data[i * m + i].abs() < EPSILON {
            rswap = i;
            break;
        }
    }
    return (rswap, m - 1);
}

fn row_swap(i: usize, j: usize, matrix: &mut NdArray) {
    debug_assert!(matrix.dims.len() == 2);
    debug_assert!(matrix.dims[0] == matrix.dims[1]);
    let m = matrix.dims[1];
    if i < usize::MAX && j < usize::MAX {
        for k in 0..m {
            matrix.data.swap(i * m + k, j * m + k);
        }
    }
}

fn normalize(v: &mut Vec<f32>) {
    let mut norm = 0_f32;
    for i in 0..v.len() {
        norm += v[i] * v[i];
    }
    norm = norm.sqrt();
    for i in 0..v.len() {
        v[i] /= norm;
    }
}

// TODO: Need to have row_swap be within the loop
pub fn retrieve_eigen(eig: f32, mut matrix: NdArray) -> Vec<f32> {
    // debug_assert!(matrix.dims.len() == 2);
    // debug_assert!(matrix.dims[0] == matrix.dims[1]);
    let m = matrix.dims[0];
    let mut evector = vec![0_f32; m];
    let lambda_i = scale_identity(m, eig);
    // (A - lambda I)v = 0
    // (A - lambda I) = L
    // Lv = 0
    matrix.diff(lambda_i);
    // // B is a rotation matrix
    // // B L v = 0;
    let (i, j) = target_rotation_indices(&matrix);
    row_swap(i, j, &mut matrix);
    // // QR = (B L )
    // // QRv = 0
    // // Q'Qrv = 0
    let qr = QrDecomposition::new(matrix);
    // // Rv' = 0 <-> O'v :: QRv = 0
    for i in (0..m).rev() {
        let diag = qr.triangle.data[i * m + i];
        if diag.abs() < EPSILON {
            evector[i] = 1_f32;
        } else {
            let mut sum = 0_f32;
            for j in i + 1..m {
                sum += qr.triangle.data[i * m + j] * evector[j];
            }
            evector[i] = -sum / diag;
        }
    }
    evector.swap(i, j);
    normalize(&mut evector);
    evector
}
