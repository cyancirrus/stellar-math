use crate::algebra::ndmethods::create_identity_matrix;
use crate::decomposition::qr::qr_decompose;
use crate::structure::ndarray::NdArray;

const STOP_CONDITION: f32 = 1e-6;

struct SchurDecomp {
    rotation: NdArray, // The accumulated orthogonal transformations (U for SVD)
    kernel: NdArray,   // The upper quasi-triangular matrix (Schur form)
}

impl SchurDecomp {
    pub fn new(rotation: NdArray, kernel: NdArray) -> Self {
        Self { rotation, kernel }
    }
}
fn real_schur_iteration(schur: SchurDecomp) -> SchurDecomp {
    let qr = qr_decompose(schur.kernel);
    let rotation = qr.left_multiply(schur.rotation); // RQ = Q'AQ
    let kernel = qr.triangle_rotation();
    SchurDecomp { rotation, kernel }
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

fn real_schur_decomp(kernel: NdArray) -> SchurDecomp {
    let rows = kernel.dims[0];
    let identity = create_identity_matrix(rows);
    let mut schur = SchurDecomp::new(identity, kernel);

    while real_schur_threshold(&schur.kernel) > STOP_CONDITION {
        schur = real_schur_iteration(schur);
    }
    schur
}
