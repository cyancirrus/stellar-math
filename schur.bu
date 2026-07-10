use crate::algebra::ndmethods::create_identity_matrix;
use crate::decomposition::qr::QrDecomposition;
use crate::structure::ndarray::NdArray;

const CONVERGENCE_CONDITION: f32 = 1e-6;
// const LIMIT_ITERATION: usize = 16;
// const LIMIT_ITERATION: usize = 64;
const LIMIT_ITERATION: usize = 1;
pub struct SchurDecomp {
    pub rotation: NdArray, // The current rotation
    pub kernel: NdArray,   // The upper quasi-triangular matrix (Schur form)
}

impl SchurDecomp {
    pub fn new(rotation: NdArray, kernel: NdArray) -> Self {
        Self { rotation, kernel }
    }
}

fn real_schur_iteration(mut schur: SchurDecomp) -> SchurDecomp {
    let mut qr = QrDecomposition::new(schur.kernel);
    // println!("q {:?}\nr {:?}", qr.projections, qr.triangle);
    // Apply Q to a matrix X ie (QR) -> Qx
    qr.triangle_rotation();
    qr.right_apply_q(&mut schur.rotation);
    // might want to make a thing where not needed or optional
    schur.kernel = qr.triangle;
    // println!("schur.kernel {:?}", schur.kernel);
    schur
}
fn real_schur_threshold(kernel: &NdArray) -> f32 {
    let rows = kernel.dims[0];
    let cols = kernel.dims[1];
    let mut off_diagonal = 0f32;

    for j in 0..cols {
        for i in j + 1..rows {
            off_diagonal += kernel.data[i * rows + j].abs();
        }
    }
    off_diagonal
}
pub fn real_schur(kernel: NdArray) -> SchurDecomp {
    let rows = kernel.dims[0];
    let rotation = create_identity_matrix(rows);
    let mut schur = SchurDecomp::new(rotation, kernel);
    for _ in 0..LIMIT_ITERATION {
        let threshold = real_schur_threshold(&schur.kernel);
        if CONVERGENCE_CONDITION > threshold {
            break;
        }
        schur = real_schur_iteration(schur);
    }
    schur
}
