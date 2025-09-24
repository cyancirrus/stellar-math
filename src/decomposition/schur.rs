use crate::algebra::ndmethods::create_identity_matrix;
use crate::decomposition::qr::{qr_decompose, QrDecomposition};
use crate::algebra::ndmethods::tensor_mult;
use crate::structure::ndarray::NdArray;

// const CONVERGENCE_CONDITION: f32 = 1e-6;
const CONVERGENCE_CONDITION: f32 = 1e-6;

pub struct SchurDecomp {
    pub rotation: NdArray, // The accumulated orthogonal transformations (U for SVD)
    pub kernel: NdArray,   // The upper quasi-triangular matrix (Schur form)
}

impl SchurDecomp {
    pub fn new(rotation: NdArray, kernel: NdArray) -> Self {
        Self { rotation, kernel }
    }
}

fn real_schur_iteration(mut schur: SchurDecomp) -> SchurDecomp {
    // Apply Q to a matrix X ie (QR) -> Qx
    let mut qr = qr_decompose(schur.kernel);
    qr.triangle_rotation(); 
    qr.left_multiply(&mut schur.rotation);
    schur.kernel = qr.triangle;
    schur
}

fn real_schur_threshold(kernel: &NdArray) -> f32 {
    let rows = kernel.dims[0];
    let cols = kernel.dims[1];
    let mut off_diagonal = 0_f32;

    for j in 0..cols {
        for i in j + 1..rows {
            off_diagonal += kernel.data[i * rows + j].abs();
        }
    }
    off_diagonal
}

pub fn real_schur(kernel: NdArray) -> SchurDecomp {
    let rows = kernel.dims[0];
    let identity = create_identity_matrix(rows);
    let mut schur = SchurDecomp::new(identity, kernel);
    println!("hello world");
    while CONVERGENCE_CONDITION < real_schur_threshold(&schur.kernel) {
        real_schur_threshold(&schur.kernel);
        println!("going");
        schur = real_schur_iteration(schur);
    }
    schur
}
