use crate::algebra::ndmethods::create_identity_matrix;
use crate::decomposition::lq::AutumnDecomp;
use crate::structure::ndarray::NdArray;

const CONVERGENCE_CONDITION: f32 = 1e-6;
const LIMIT_ITERATION: usize = 2156;
pub struct SchurDecomp {
    pub rotation: NdArray, // The current rotation
    pub kernel: NdArray,   // The upper quasi-triangular matrix (Schur form)
}

fn real_schur_iteration(
    kernel: NdArray,
    mut nkernel: NdArray,
    rotation: &mut NdArray,
    workspace: &mut [f32],
) -> (NdArray, NdArray) {
    let lq = AutumnDecomp::new(kernel);
    // println!("--------------------------------------------------");
    // println!("lq tau {:?}\nlq h {:?}", lq.t, lq.h);

    lq.mat_ql_apply(&mut nkernel, workspace);
    lq.mat_left_apply_q(rotation, workspace);
    // println!("kernel {nkernel:?}");
    (nkernel, lq.h)
}
fn real_schur_threshold(kernel: &NdArray) -> f32 {
    let rows = kernel.dims[0];
    let cols = kernel.dims[1];
    let mut off_diagonal = 0f32;

    for i in 0..rows {
        for j in i + 1..cols {
            off_diagonal += kernel.data[i * rows + j].abs();
        }
    }
    off_diagonal
}
pub fn real_schur(mut kernel: NdArray, mut nkernel: NdArray, workspace: &mut [f32]) -> SchurDecomp {
    let rows = kernel.dims[0];
    let mut rotation = create_identity_matrix(rows);
    workspace.fill(0f32);
    for _ in 0..LIMIT_ITERATION {
        let threshold = real_schur_threshold(&kernel);
        if CONVERGENCE_CONDITION > threshold {
            break;
        }
        (kernel, nkernel) = real_schur_iteration(kernel, nkernel, &mut rotation, workspace);
    }
    SchurDecomp { kernel, rotation }
}

#[cfg(test)]
mod test_lq {
    const TOLERANCE: f32 = 1e-3;
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::algebra::ndmethods::create_identity_matrix;
    use crate::equality::approximate::approx_vector_tol_eq;
    use crate::random::generation::generate_random_matrix;
    #[test]
    fn test_reconstruction() {
        for n in 1..12 {
            check_reconstruct(n);
        }
    }
    fn check_reconstruct(n: usize) {
        let x = generate_random_matrix(n, n);
        let kernel = x.clone();
        let nkernel = create_identity_matrix(n);
        let mut workspace = vec![0f32; n];
        let schur = real_schur(kernel, nkernel, &mut workspace);
        let q = schur.rotation;
        let q_star = q.transpose();
        let expect = basic_mult(&q, &x);
        let expect = basic_mult(&expect, &q_star);
        let result = schur.kernel;
        println!("expect {expect:?}");
        println!("result {result:?}");
        assert!(approx_vector_tol_eq(&result.data, &expect.data, TOLERANCE));
    }
    #[test]
    fn test_orthogonalality() {
        for n in 1..12 {
            check_orthogonal(n);
        }
    }
    fn check_orthogonal(n: usize) {
        let x = generate_random_matrix(n, n);
        let kernel = x.clone();
        let nkernel = create_identity_matrix(n);
        let mut workspace = vec![0f32; n];
        let schur = real_schur(kernel, nkernel, &mut workspace);
        let q = schur.rotation;
        let q_star = q.transpose();
        let expect = create_identity_matrix(n);
        let result = basic_mult(&q, &q_star);
        println!("expect {expect:?}");
        println!("result {result:?}");
        assert!(approx_vector_tol_eq(&result.data, &expect.data, TOLERANCE));
    }
}
